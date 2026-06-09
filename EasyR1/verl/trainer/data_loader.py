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

import os
from typing import Optional

import torch
from torch.utils.data import BatchSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig


class ModalityAwareBatchSampler(BatchSampler):
    def __init__(self, dataset: RLHFDataset, batch_size: int, shuffle: bool, seed: int, data_type_key: str = "data_type"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.data_type_key = data_type_key

        self.video_indices = []
        self.image_indices = []

        raw_data_types = dataset.dataset[data_type_key]
        for idx, data_type in enumerate(raw_data_types):
            normalized_data_type = str(data_type).strip().lower()
            if normalized_data_type == "video":
                self.video_indices.append(idx)
            elif normalized_data_type == "image":
                self.image_indices.append(idx)
            else:
                raise ValueError(
                    f"Unsupported data_type '{data_type}' at index={idx}. "
                    "Expected one of ['video', 'image']."
                )

        self.video_steps = len(self.video_indices) // self.batch_size
        self.image_steps = len(self.image_indices) // self.batch_size
        self.total_steps = self.video_steps + self.image_steps

        if self.total_steps <= 0:
            raise ValueError(
                "No full batches can be formed. "
                f"video_samples={len(self.video_indices)}, image_samples={len(self.image_indices)}, batch_size={self.batch_size}."
            )

        self.video_drop_count = len(self.video_indices) - self.video_steps * self.batch_size
        self.image_drop_count = len(self.image_indices) - self.image_steps * self.batch_size

        self._epoch = 0
        self._consumed_batches = 0

    def __len__(self) -> int:
        return self.total_steps

    def _build_epoch_batches(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)

        def _prepare_batches(indices: list[int], steps: int):
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=generator).tolist()
                ordered_indices = [indices[i] for i in perm]
            else:
                ordered_indices = list(indices)

            return [
                ordered_indices[i * self.batch_size : (i + 1) * self.batch_size]
                for i in range(steps)
            ]

        video_batches = _prepare_batches(self.video_indices, self.video_steps)
        image_batches = _prepare_batches(self.image_indices, self.image_steps)

        batch_plan = (["video"] * self.video_steps) + (["image"] * self.image_steps)
        if self.shuffle:
            plan_perm = torch.randperm(len(batch_plan), generator=generator).tolist()
            batch_plan = [batch_plan[i] for i in plan_perm]

        return video_batches, image_batches, batch_plan

    def __iter__(self):
        video_batches, image_batches, batch_plan = self._build_epoch_batches()

        video_ptr = 0
        image_ptr = 0

        for modality in batch_plan[: self._consumed_batches]:
            if modality == "video":
                video_ptr += 1
            else:
                image_ptr += 1

        for modality in batch_plan[self._consumed_batches :]:
            if modality == "video":
                batch = video_batches[video_ptr]
                video_ptr += 1
            else:
                batch = image_batches[image_ptr]
                image_ptr += 1

            self._consumed_batches += 1
            yield batch

        self._epoch += 1
        self._consumed_batches = 0

    def state_dict(self) -> dict:
        return {
            "epoch": self._epoch,
            "consumed_batches": self._consumed_batches,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._epoch = int(state_dict.get("epoch", 0))
        self._consumed_batches = int(state_dict.get("consumed_batches", 0))



def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,

        ### Embodied-R1.5 New Feature ###
        problem_type_key=config.problem_type_key,
        problem_id_key=config.problem_id_key,
        options_key=config.options_key,
        data_type_key=config.data_type_key,
        data_source_key=config.data_source_key,
        ### Embodied-R1.5 New Feature ###

        image_dir=config.image_dir,
        max_frames=config.max_frames,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
        video_fps=config.video_fps,
    )
    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_batch_sampler = ModalityAwareBatchSampler(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=config.shuffle,
        seed=config.seed,
        data_type_key=config.data_type_key,
    )

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    total_train_samples = len(train_dataset)
    print(
        "Train modality sampler stats: "
        f"N_video={len(train_batch_sampler.video_indices)}, "
        f"N_image={len(train_batch_sampler.image_indices)}, "
        f"S_video={train_batch_sampler.video_steps}, "
        f"S_image={train_batch_sampler.image_steps}, "
        f"drop_video={train_batch_sampler.video_drop_count}, "
        f"drop_image={train_batch_sampler.image_drop_count}, "
        f"used_steps={len(train_batch_sampler)}, "
        f"used_samples={len(train_batch_sampler) * train_batch_size}/{total_train_samples}"
    )

    val_dataset = RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,

        ### Embodied-R1.5 New Feature ###
        problem_type_key=config.problem_type_key,
        problem_id_key=config.problem_id_key,
        options_key=config.options_key,
        data_type_key=config.data_type_key,
        data_source_key=config.data_source_key,
        ### Embodied-R1.5 New Feature ###

        image_dir=config.image_dir,
        max_frames=config.max_frames,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        video_fps=config.video_fps,
    )

    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
