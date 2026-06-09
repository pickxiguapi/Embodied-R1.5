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

import importlib.util
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


### Embodied-R1.5 New Feature ###
class RewardInputRequired(TypedDict):
    """Required fields for all reward functions"""
    response: str
    response_length: int
    ground_truth: str

class RewardInput(RewardInputRequired, total=False):
    """
    Optional fields for Embodied-R1.5 multi-task data format.
    """
    options: list
    problem_type: str
    problem_id: int
    data_type: str
    data_source: str
    problem: str
    dataset_name: str
### Embodied-R1.5 New Feature ###


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]
    dataset_name: Optional[str]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class SequentialFunctionRewardManagerMixin:
    reward_fn: SequentialRewardFunction

    def compute_reward_sequential(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        dataset_names = []  # Store dataset_name separately
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )

            # Debug: print first 3 responses
            # if i < 3:
            #     print(f"[DEBUG] Sample {i} response: {response_str[:500]}...")
            #     print(f"[DEBUG] Sample {i} ground_truth: {data.non_tensor_batch['ground_truth'][i]}")

            ### Embodied-R1.5 New Feature ###
            reward_input = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            # Add optional fields if they exist in non_tensor_batch
            optional_fields = ["options", "problem_type", "problem_id", "data_type", "data_source", "problem", "dataset_name"]
            for field in optional_fields:
                if field in data.non_tensor_batch:
                    reward_input[field] = data.non_tensor_batch[field][i]
            ### Embodied-R1.5 New Feature ###

            score = self.reward_fn(reward_input)
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                if key == "dataset_name":
                    dataset_names.append(value)  # Store separately, don't add to reward_metrics
                else:
                    reward_metrics[key].append(value)

        # Add dataset-specific metrics
        if dataset_names:
            dataset_metrics = defaultdict(lambda: defaultdict(list))
            for i, dataset_name in enumerate(dataset_names):
                if dataset_name is not None:
                    for key in ["overall", "format", "accuracy"]:
                        if key in reward_metrics:
                            dataset_metrics[dataset_name][key].append(reward_metrics[key][i])

            # Add grouped metrics to reward_metrics
            for dataset_name, metrics in dataset_metrics.items():
                for metric_name, values in metrics.items():
                    reward_metrics[f"{dataset_name}/{metric_name}"] = values

        return reward_tensor, reward_metrics


class BatchFunctionRewardManagerMixin:
    reward_fn: BatchRewardFunction

    def compute_reward_batch(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )

            # Debug: print first 5 samples with full content
            # if i < 5:
            #     print(f"\n{'='*50}")
            #     print(f"[DEBUG-BATCH] Sample {i}")
            #     print(f"[DEBUG-BATCH] FULL RESPONSE:\n{response_str}")
            #     print(f"[DEBUG-BATCH] GROUND_TRUTH: {data.non_tensor_batch['ground_truth'][i]}")
            #     problem = data.non_tensor_batch.get('problem', [None])[i]
            #     print(f"[DEBUG-BATCH] FULL PROBLEM:\n{problem if problem else 'N/A'}")
            #     print(f"{'='*50}\n")

            ### Embodied-R1.5 New Feature ###
            reward_input = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            # Add optional fields if they exist in non_tensor_batch
            optional_fields = ["options", "problem_type", "problem_id", "data_type", "data_source", "problem", "dataset_name"]
            for field in optional_fields:
                if field in data.non_tensor_batch:
                    reward_input[field] = data.non_tensor_batch[field][i]
            ### Embodied-R1.5 New Feature ###

            reward_inputs.append(reward_input)

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        dataset_names = []  # Store dataset_name separately
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                if key == "dataset_name":
                    dataset_names.append(value)  # Store separately, don't add to reward_metrics
                else:
                    reward_metrics[key].append(value)

        # Add dataset-specific metrics
        if dataset_names:
            dataset_metrics = defaultdict(lambda: defaultdict(list))
            for i, dataset_name in enumerate(dataset_names):
                if dataset_name is not None:
                    for key in ["overall", "format", "accuracy"]:
                        if key in reward_metrics:
                            dataset_metrics[dataset_name][key].append(reward_metrics[key][i])

            # Add grouped metrics to reward_metrics
            for dataset_name, metrics in dataset_metrics.items():
                for metric_name, values in metrics.items():
                    reward_metrics[f"{dataset_name}/{metric_name}"] = values

        return reward_tensor, reward_metrics


class AutoRewardManager(BatchFunctionRewardManagerMixin, SequentialFunctionRewardManagerMixin):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        reward_name = getattr(module, "REWARD_NAME", "unknown")
        reward_type = getattr(module, "REWARD_TYPE", "batch")
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        print(f"Reward name: {reward_name}, reward type: {reward_type}.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.reward_type = reward_type
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        if self.reward_type == "batch":
            return self.compute_reward_batch(data)
        elif self.reward_type == "sequential":
            return self.compute_reward_sequential(data)
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}.")
