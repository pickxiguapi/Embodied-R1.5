# vllm serve command for Embodied-R1.5
"""

vllm serve IffYuan/Embodied-R1.5 \
  --served-model-name "Embodied-R1.5" \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --gpu-memory-utilization 0.7 \
  --async-scheduling \
  --media-io-kwargs '{"video": {"num_frames": 32}, "image": {"max_num": 32}}' \
  --max_model_len 20000 \
  --limit-mm-per-prompt '{"image": 8, "video": 1}' \
  --host 0.0.0.0 \
  --port 22002

# nohup version
nohup vllm serve IffYuan/Embodied-R1.5 \
  --served-model-name "Embodied-R1.5" \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --gpu-memory-utilization 0.7 \
  --async-scheduling \
  --media-io-kwargs '{"video": {"num_frames": 32}, "image": {"max_num": 32}}' \
  --max_model_len 20000 \
  --limit-mm-per-prompt '{"image": 8, "video": 1}' \
  --host 0.0.0.0 \
  --port 22002 > vllm.log 2>&1 &

"""

import base64
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Union

from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please answer this question based on the visual content."
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer."
    "At the end, you must output the final answer in the format:\n"
    "<answer><your_answer_here></answer>\n"
)

class VLLMOnlineClient:
    """Client for calling vLLM online API with support for images and videos"""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:22002/v1",
        api_key: str = "EMPTY",
        timeout: int = 3600,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model_name = model_name
        logger.info(f"Client initialized: {model_name}")

    @staticmethod
    def encode_image(image_path: Union[str, Path]) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')

        image_path = Path(image_path)
        suffix = image_path.suffix.lower()
        mime_type = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
            '.gif': 'image/gif', '.webp': 'image/webp'
        }.get(suffix, 'image/jpeg')

        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def encode_video(video_path: Union[str, Path]) -> str:
        """Encode video to base64 string"""
        with open(video_path, "rb") as video_file:
            encoded = base64.b64encode(video_file.read()).decode('utf-8')

        video_path = Path(video_path)
        suffix = video_path.suffix.lower()
        mime_type = {
            '.mp4': 'video/mp4', '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime', '.mkv': 'video/x-matroska', '.webm': 'video/webm'
        }.get(suffix, 'video/mp4')

        return f"data:{mime_type};base64,{encoded}"

    def prepare_messages(
        self,
        test_case: Dict[str, Any],
        base_path: str = "",
    ) -> List[Dict[str, Any]]:
        """Build messages from test case data"""
        content = []
        prompt_text = test_case["prompt"]
        if QUESTION_TEMPLATE:
            prompt_text = QUESTION_TEMPLATE.format(Question=test_case["prompt"])
        test_type = test_case["type"]

        if test_type == "single_image":
            image_path = Path(base_path) / test_case["image"]
            encoded_image = self.encode_image(image_path)
            content.append({"type": "image_url", "image_url": {"url": encoded_image}})
            content.append({"type": "text", "text": prompt_text})

        elif test_type == "multi_image":
            for img_path in test_case["image"]:
                full_path = Path(base_path) / img_path
                encoded_image = self.encode_image(full_path)
                content.append({"type": "image_url", "image_url": {"url": encoded_image}})
            content.append({"type": "text", "text": prompt_text})

        elif test_type == "video":
            video_data = test_case["video"]
            valid_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            for vid_path in video_data:
                full_path = Path(base_path) / vid_path
                if full_path.suffix.lower() not in valid_video_formats:
                    raise ValueError(f"Invalid video format: {full_path.suffix}. Supported formats: {', '.join(valid_video_formats)}")
                encoded_video = self.encode_video(full_path)
                content.append({"type": "video_url", "video_url": {"url": encoded_video}})
            content.append({"type": "text", "text": prompt_text})

        return [{"role": "user", "content": content}]


if __name__ == "__main__":
    # Model and server configuration
    base_url = "http://localhost:22002/v1"
    api_key = "EMPTY"
    model_name = "Embodied-R1.5"

    # Sampling parameters
    seed = 3407
    top_p = 0.8
    temperature = 0.7
    presence_penalty = 1.5
    max_tokens = 4096  # out_seq_length

    # Video processing parameters (for mm_processor_kwargs)
    video_fps = 2  # Frames per second for video sampling
    video_do_sample_frames = True  # Enable frame sampling

    # Initialize client
    client = VLLMOnlineClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        timeout=3600
    )

    # Test Case 0
    print('\n' + '=' * 80)
    print("Case 0 - Type: single_image")
    print('=' * 80)

    case_0 = {
        "idx": 0,
        "answer": "(C)",
        "prompt": "How many table lamps are in the image? Select from the following choices.\n(A) 0\n(B) 2\n(C) 1\n(D) 3",
        "image": "test_assets/sample_2_image.png",
        "video": "",
        "type": "single_image"
    }

    messages = client.prepare_messages(case_0)
    start_time = time.time()

    response = client.client.chat.completions.create(
        model=client.model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    latency = time.time() - start_time
    generated_text = response.choices[0].message.content
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    tokens = usage.total_tokens if usage else 0
    
    print(f"\nPrompt: {case_0['prompt']}")
    print(f"\nGenerated Answer: {generated_text}")
    print(f"Latency: {latency:.2f}s | Tokens: {prompt_tokens}+{completion_tokens}={tokens}")
    print('-' * 80)

    # Test Case 1
    print('\n' + '=' * 80)
    print("Case 1 - Type: single_image")
    print('=' * 80)

    case_1 = {
        "idx": 1,
        "answer": "door",
        "prompt": "Among the listed objects, which one is closest to your current location in the image? The option: table, towel, door or basket. Direct answer the question",
        "image": "test_assets/sample_0_image.png",
        "video": "",
        "type": "single_image"
    }

    messages = client.prepare_messages(case_1)
    start_time = time.time()

    response = client.client.chat.completions.create(
        model=client.model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    latency = time.time() - start_time
    generated_text = response.choices[0].message.content
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    tokens = usage.total_tokens if usage else 0

    print(f"\nPrompt: {case_1['prompt']}")
    print(f"\nGenerated Answer: {generated_text}")
    print(f"Latency: {latency:.2f}s | Tokens: {prompt_tokens}+{completion_tokens}={tokens}")
    print('-' * 80)

    # Test Case 2
    print('\n' + '=' * 80)
    print("Case 2 - Type: single_image")
    print('=' * 80)

    case_2 = {
        "idx": 2,
        "answer": "",
        "prompt": "Provide one or more points coordinate of objects region this sentence describes: you need to grasp the mug. The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}].",
        "image": "test_assets/aff.png",
        "video": "",
        "type": "single_image"
    }

    messages = client.prepare_messages(case_2)
    start_time = time.time()

    response = client.client.chat.completions.create(
        model=client.model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    latency = time.time() - start_time
    generated_text = response.choices[0].message.content
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    tokens = usage.total_tokens if usage else 0

    print(f"\nPrompt: {case_2['prompt']}")
    print(f"\nGenerated Answer: {generated_text}")
    print(f"Latency: {latency:.2f}s | Tokens: {prompt_tokens}+{completion_tokens}={tokens}")
    print('-' * 80)

    # Test Case 3
    print('\n' + '=' * 80)
    print("Case 3 - Type: multi_image")
    print('=' * 80)

    case_3 = {
        "idx": 3,
        "answer": "place sticky notes in the stand",
        "prompt": "current goal is: please stock caddy for phone room. last 20 steps: 1- place black pen in pen stand 2- place black pen in pen stand 3- place black pen in pen stand 4- place black pen in pen stand 5- place blue pen in pen stand 6- place blue pen in pen stand 7- place blue pen in pen stand. what is the immediate next step?",
        "image": [
            "test_assets/robovqa_086678/01.jpg",
            "test_assets/robovqa_086678/02.jpg",
            "test_assets/robovqa_086678/03.jpg",
            "test_assets/robovqa_086678/04.jpg",
            "test_assets/robovqa_086678/05.jpg",
            "test_assets/robovqa_086678/06.jpg",
            "test_assets/robovqa_086678/07.jpg",
            "test_assets/robovqa_086678/08.jpg"
        ],
        "video": "",
        "type": "multi_image"
    }

    messages = client.prepare_messages(case_3)
    start_time = time.time()

    response = client.client.chat.completions.create(
        model=client.model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    latency = time.time() - start_time
    generated_text = response.choices[0].message.content
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    tokens = usage.total_tokens if usage else 0

    print(f"\nPrompt: {case_3['prompt']}")
    print(f"\nGenerated Answer: {generated_text}")
    print(f"Latency: {latency:.2f}s | Tokens: {prompt_tokens}+{completion_tokens}={tokens}")
    print('-' * 80)

    # Test Case 4
    print('\n' + '=' * 80)
    print("Case 4 - Type: video")
    print('=' * 80)

    case_4 = {
        "idx": 5,
        "answer": "",
        "prompt": "\nThese are frames of a video.\nIf I am standing by the trash can and facing the counter top, is the dog bed to the left or the right of the counter top?\nA. left\nB. right\nAnswer with the option's letter from the given choices directly.",
        "image": "",
        "video": [
            "test_assets/5a83c3ab-9a60-4c9e-93be-4735d522f3f1.mp4"
        ],
        "type": "video"
    }

    messages = client.prepare_messages(case_4)
    start_time = time.time()

    response = client.client.chat.completions.create(
        model=client.model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    latency = time.time() - start_time
    generated_text = response.choices[0].message.content
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    tokens = usage.total_tokens if usage else 0

    print(f"\nPrompt: {case_4['prompt']}")
    print(f"\nGenerated Answer: {generated_text}")
    print(f"Latency: {latency:.2f}s | Tokens: {prompt_tokens}+{completion_tokens}={tokens}")
    print('-' * 80)
