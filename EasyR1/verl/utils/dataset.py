# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import uuid
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import Features, Sequence, Value, concatenate_datasets, load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF
from .qwen_vl_utils.vision_process_official import process_vision_info


QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please answer this question based on the visual content."
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer."
    "At the end, you must output the final answer in the format:\n"
    "<answer><your_answer_here></answer>\n"
)

VIDEO_CACHE_DIR = os.environ.get("ER15_VIDEO_CACHE_DIR", os.path.join(os.getcwd(), ".er1_5_video_cache"))
VIDEO_CACHE_TAG = "video_cache_v1_fps_2_frame_32"
VIDEO_CACHE_WRITE = True

TYPE_TEMPLATE = {
    "multiple choice": (
        "Please provide only the single option letter (e.g., A, B, C, D, etc.) "
        "within the <answer>...</answer> tags.\n"
        "Example:\n<answer>A</answer>"
    ),
    "numerical": (
        "Please provide only the numerical value within the <answer>...</answer> tags.\n"
        "Example:\n<answer>3.14</answer>"
    ),
    "open-ended": (
        "Please provide only your text answer within the <answer>...</answer> tags.\n"
        "Example:\n<answer>The capital of France is Paris.</answer>"
    ),
    "math": (
        "Please provide only the final result (a number or LaTeX formula) within the <answer>...</answer> tags.\n"
        "Example:\n<answer>$$-\\dfrac{3}{2}$$</answer>"
    ),
    "spatial grounding": (
        "Please provide only the bounding box as JSON with key 'boxes' within the <answer>...</answer> tags.\n"
        "Example:\n<answer>{\"boxes\": [35, 227, 437, 932]}</answer>"
    ),
    "trace": (
        "Please provide only the ordered waypoints as JSON with key 'point_2d' within the <answer>...</answer> tags.\n"
        "Example:\n<answer>```json\n[{\"point_2d\": [624, 469]}, {\"point_2d\": [640, 421]}, {\"point_2d\": [638, 372]}, {\"point_2d\": [613, 337]}]\n```</answer>"
    ),
    "trace_3d": (
        "Please provide only the ordered 2D waypoints with depth (in meters) as JSON with key 'point_2d' and 'depth' within the <answer>...</answer> tags.\n"
        "Example:\n<answer>```json\n[{\"point_2d\": [463, 599], \"depth\": 1.08}, {\"point_2d\": [458, 603], \"depth\": 1.08}, {\"point_2d\": [449, 612], \"depth\": 1.06}]\n```</answer>"
    ),
    "point": (
        "Please pointing to answer the question within the <answer>...</answer> tags.\n"
        "Example:\n<answer>```json\n[{\"point_2d\": [230, 138]}]\n```</answer>"
    ),
}


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        # original_pixels = image.width * image.height
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        # new_pixels = width * height
        #print(f"[Image Resize] max_pixels triggered: {original_pixels} pixels ({image.width}x{image.height}) -> {new_pixels} pixels ({width}x{height})")
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        # original_pixels = image.width * image.height
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        # new_pixels = width * height
        #print(f"[Image Resize] min_pixels triggered: {original_pixels} pixels ({image.width}x{image.height}) -> {new_pixels} pixels ({width}x{height})")
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: Union[str, list[str], dict[str, str]],  ### Embodied-R1.5: Support multiple files and dict format ###
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "problem",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",

        ### Embodied-R1.5 New Feature ###
        problem_type_key: str = "problem_type",
        problem_id_key: str = "problem_id",
        options_key: str = "options",
        data_type_key: str = "data_type",
        data_source_key: str = "data_source",
        ### Embodied-R1.5 New Feature ###

        image_dir: Optional[str] = None,
        max_frames: int = 32,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        video_fps: float = 2.0,
        debug: bool = False,  # Debug mode: sample 200 examples per dataset
        debug_sample_size: int = 200,  # Number of samples in debug mode
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key

        ### Embodied-R1.5 New Feature ###
        self.problem_type_key = problem_type_key
        self.problem_id_key = problem_id_key
        self.options_key = options_key
        self.data_type_key = data_type_key
        self.data_source_key = data_source_key
        ### Embodied-R1.5 New Feature ###

        self.image_dir = image_dir
        self.max_frames = max_frames
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.video_fps = video_fps
        self.debug = debug
        self.debug_sample_size = debug_sample_size

        ### Embodied-R1.5 New Feature: Support multiple data files and dict format ###
        # Helper function to extract dataset_name from file path
        def extract_dataset_name(file_path: str) -> str:
            basename = os.path.basename(file_path)
            dataset_name = os.path.splitext(basename)[0]
            return dataset_name

        if isinstance(data_path, dict):
            # Dict format: {dataset_name: path}
            data_items = list(data_path.items())
        elif isinstance(data_path, str):
            # Single file: extract dataset_name from filename
            data_items = [(extract_dataset_name(data_path), data_path)]
        else:
            # List format: extract dataset_name from each filename
            data_items = [(extract_dataset_name(path), path) for path in data_path]

        datasets = []
        for dataset_name, single_path in data_items:
            print(f"Processing dataset: {dataset_name}")

            # todo: hard code
            DATASET_CONFIG = {
                "ER1.5_general_video_qa_50s_cleaned": 7311,
                "ER1.5_Cosmos_video_qa": 700,
                "ER1.5_CoSyn-point_image_point": 5000,
                "ER1.5_Droid-Trace_image_trace": 13000,
                "ER1.5_EgoPlan_mixed_qa": 9919,
                "ER1.5_EO_image_qa": 5000,
                "ER1.5_ER1-point_image_point": 20000,
                "ER1.5_ER1-trace_image_trace": 15000,
                "ER1.5_ERQA2_image_qa": 5000,
                "ER1.5_ERQA_Rush_image_qa": 310,
                "ER1.5_general_image_qa_filtered": 20000,
                "ER1.5_general_video_qa": 10000,
                "ER1.5_HandAL_image_point": 20000,
                "ER1.5_HOI4D-Trace_image_trace": 4000,
                "ER1.5_InstructPart_image_point": 3546,
                "ER1.5_InternData-Trace_image_trace": 5000,
                "ER1.5_Ref_L4_image_point": 5000,
                "ER1.5_Refspatial_image_point": 15000,
                "ER1.5_regular_simulation_image_point": 2000,
                "ER1.5_regular_synthetic_image_point": 3000,
                "ER1.5_Robo2VLM_image_qa": 5000,
                "ER1.5_robocasa_partnet_2d_image_trace": 1000,
                "ER1.5_robocasa_partnet_3d_image_trace": 1000,
                "ER1.5_Roborefit_image_point": 8000,
                "ER1.5_RoboVQA_image": 10000,
                "ER1.5_SAT_image_qa": 5000,
                "ER1.5_spatialssrl_image_qa": 10000,
                "ER1.5_EmbSpatial_image_qa": 5099,
            }

            data_split = "train"

            # Load dataset: support local JSON files or remote HuggingFace datasets
            if os.path.isfile(single_path):
                # Local JSON file
                ds = load_dataset("json", data_files=single_path, split=data_split)
            else:
                # Remote dataset from HuggingFace Hub
                ds = load_dataset(single_path, split=data_split)

            # Add dataset_name field
            ds = ds.map(lambda x: {**x, "dataset_name": dataset_name}, desc=f"Adding dataset_name: {dataset_name}")

            # Debug mode: randomly sample examples
            if self.debug:
                original_size = len(ds)
                sample_size = min(self.debug_sample_size, original_size)
                ds = ds.shuffle(seed=42).select(range(sample_size))
                print(f"  [DEBUG MODE] Sampled {sample_size}/{original_size} examples from {dataset_name}")

            # Sample based on DATASET_CONFIG
            if dataset_name in DATASET_CONFIG:
                max_num = DATASET_CONFIG[dataset_name]
                original_size = len(ds)
                if original_size > max_num:
                    ds = ds.shuffle(seed=42).select(range(max_num))
                    print(f"  [SAMPLING] Sampled {max_num}/{original_size} examples from {dataset_name}")
                else:
                    print(f"  [SAMPLING] Using all {original_size} examples from {dataset_name} (max_num={max_num})")
            else:
                print(f"  [WARNING] {dataset_name} not in DATASET_CONFIG, using all examples")

            datasets.append(ds)

        # Concatenate all datasets if multiple files provided
        common_features = Features({
            'problem_id': Value('string'),
            'problem': Value('string'),
            'data_type': Value('string'),
            'problem_type': Value('string'),
            'options': Sequence(Value('string')),
            'data_source': Value('string'),
            'answer': Value('string'),
            'problem_reserved_text': Value('string'),
            'images': Sequence(Value('string')),
            'videos': Sequence(Value('string')),
            'dataset_name': Value('string'),
        })

        # Add missing fields
        aligned_datasets = []
        for ds in datasets:
            # Add missing 'images' field if not present
            if 'images' not in ds.column_names:
                ds = ds.map(lambda x: {**x, 'images': []})
            # Add missing 'videos' field if not present
            if 'videos' not in ds.column_names:
                ds = ds.map(lambda x: {**x, 'videos': []})

            # Cast to common schema
            aligned_datasets.append(ds.cast(common_features))

        if len(datasets) == 1:
            self.dataset = aligned_datasets[0]
        else:
            self.dataset = concatenate_datasets(aligned_datasets)
            print(f"Loaded and concatenated {len(datasets)} datasets, total samples: {len(self.dataset)}")

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Build messages for the example.

        Args:
            example: The example dict
        """
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        data_type = example.get(self.data_type_key, "").strip().lower()
        problem_type = example.get(self.problem_type_key, "")

        question = prompt_str
        if (problem_type == "multiple choice") and isinstance(example.get("options"), list) and example["options"]:
            opts = "\n".join(example["options"])
            question = f"{prompt_str}\nOptions:\n{opts}"

        tail = TYPE_TEMPLATE.get(problem_type, "")
        prompt_str = QUESTION_TEMPLATE.format(Question=question) + tail

        if data_type == "image":
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif data_type == "video":
            if prompt_str.count("<video>") != 1:
                raise ValueError(
                    f"Video sample expects exactly one <video> placeholder, but got {prompt_str.count('<video>')}"
                )

            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        """Filter out examples with prompts longer than max_prompt_length."""
        messages = self._build_messages(example)
        data_type = example.get(self.data_type_key, "").strip().lower()

        if data_type == "image":
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.get(self.image_key, [])
            if self.image_dir and images and isinstance(images[0], str):
                images = [os.path.join(self.image_dir, img) for img in images]
            processed_images = [process_image(img, self.min_pixels, self.max_pixels) for img in images] or None
            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif data_type == "video":
            return True  # Skip filtering for video
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def _get_video_cache_path(self, dataset_name: str, problem_id: str) -> Optional[str]:
        cache_path = os.path.join(VIDEO_CACHE_DIR, VIDEO_CACHE_TAG, dataset_name, f"{problem_id}.pt")
        return cache_path

    def _atomic_torch_save(self, obj: Any, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp.{uuid.uuid4().hex}"
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        data_type = example.get(self.data_type_key, "").strip().lower()
        example["multi_modal_data"] = None
        example["mm_processor_kwargs"] = None

        if data_type == "image":
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif data_type == "video":
            assert self.processor is not None
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            dataset_name = example.get("dataset_name", None)
            problem_id = example.get(self.problem_id_key, None)
            if dataset_name is None or problem_id is None:
                raise KeyError(
                    f"Video cache key requires 'dataset_name' and '{self.problem_id_key}', got dataset_name={dataset_name}, {self.problem_id_key}={problem_id}"
                )

            cache_path = self._get_video_cache_path(dataset_name=dataset_name, problem_id=problem_id)
            # print("cache path: ", cache_path)
            videos = None
            video_kwargs = None
            cache_payload_ok = False
            allow_write_cache = VIDEO_CACHE_WRITE

            if cache_path is not None and os.path.exists(cache_path):
                try:
                    payload = torch.load(cache_path, map_location="cpu")
                    videos = payload.get("videos", None)
                    video_kwargs = payload.get("video_kwargs", None)
                    cache_payload_ok = (videos is not None) and (video_kwargs is not None)
                except Exception:
                    # cache 文件可能损坏/不兼容：忽略，走 decode
                    videos = None
                    video_kwargs = None
                    cache_payload_ok = False
                    print("Missing!!!!!!")
            else:
                videos = None
                video_kwargs = None
                print(cache_path)
                print("Missing2!!!!!!")

            if videos is None or video_kwargs is None:
                video_paths = example.pop(self.video_key)
                if self.image_dir is not None and len(video_paths) != 0 and isinstance(video_paths[0], str):  # video paths
                    video_paths = [os.path.join(self.image_dir, v) for v in video_paths]
                assert len(video_paths) == 1
                video_path = video_paths[0]
                for msg in messages:
                    if isinstance(msg.get("content"), list):
                        for part in msg["content"]:
                            if isinstance(part, dict) and part.get("type") == "video" and "video" not in part:
                                part["video"] = video_path
                                part["fps"] = self.video_fps
                                part["max_frames"] = self.max_frames
                                part["min_pixels"] = 8 * 32 * 32
                                part["max_pixels"] = 100 * 32 * 32

                image_patch_size = 16
                _, videos, video_kwargs = process_vision_info(
                    messages,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                    image_patch_size=image_patch_size,
                )

                if allow_write_cache and cache_path is not None and (not cache_payload_ok):
                    self._atomic_torch_save({"videos": videos, "video_kwargs": video_kwargs}, cache_path)

            videos_for_processor = [v[0] if isinstance(v, tuple) else v for v in videos]

            model_inputs = self.processor(
                text=prompt,
                images=None,
                videos=videos_for_processor,
                padding=True,
                return_tensors="pt",
            )

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
            example["mm_processor_kwargs"] = video_kwargs
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if "images" in example:
            example.pop("images")
        if "videos" in example:
            example.pop("videos")
        if "options" in example:
            example.pop("options")

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw", None),
                    video_grid_thw=model_inputs.get("video_grid_thw", None),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                    attention_mask=attention_mask,
                )  # (3, seq_length)
                text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
            else:
                raise ValueError("Must be Qwen3-VL")
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)


        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)

        # Map problem_reserved_text to problem for reward function
        if "problem_reserved_text" in example:
            example["problem"] = example["problem_reserved_text"]
        return example
