import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

logger = logging.getLogger(__name__)


class VLLMInferenceEngine:
    """vLLM inference engine wrapper for multimodal models (local models only)"""
    
    def __init__(
        self,
        model_path: str,
        model_name: str,
        backbone: Optional[str] = "qwen3",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 10240,
        seed: int = 3407,
        limit_mm_per_prompt: dict = None,
    ):
        """
        Initialize vLLM inference engine
        
        Args:
            model_path: Path to the model
            model_name: Name of the model
            backbone: Backbone type ('qwen3'). 
                     If None, will auto-detect from model_name
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            max_model_len: Maximum model context length
            seed: Random seed for reproducibility
            limit_mm_per_prompt: Limit for multimodal data per prompt
        """
        self.model_path = model_path
        self.model_name = model_name
        self.backbone = backbone
        
        logger.info(f"Initializing vLLM...")
        logger.info(f"  Model path: {model_path}")
        logger.info(f"  Model name: {model_name}")
        logger.info(f"  Backbone: {self.backbone}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        logger.info(f"  Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"  Seed: {seed}")
        
        # Load processor and initialize vLLM
        if self.backbone in ['qwen3']:
            self.processor = AutoProcessor.from_pretrained(model_path)
            logger.info(f"✓ Processor loaded")
            
            # Initialize vLLM
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=max_model_len,
                limit_mm_per_prompt=limit_mm_per_prompt,
                seed=seed,
            )
            logger.info(f"✓ vLLM initialized\n")
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
    
    def prepare_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare message format for a single sample, supporting multiple images and videos"""
        question = sample['question']

        # Handle options if present
        if isinstance(sample.get("options"), list) and sample["options"]:
            opts = "\n".join(sample["options"])
            question = f"{question}\nOptions:\n{opts}"
        print(question)

        # Build content list
        content = []
        
        # Handle images (single or multiple)
        if 'image' in sample and sample['image'] is not None:
            images = sample['image']
            # Convert single image to list for uniform processing
            if not isinstance(images, list):
                images = [images]
            
            for img in images:
                if img is not None:
                    content.append({"type": "image", "image": img})
        
        # Handle videos (single or multiple mp4 files)
        if 'video' in sample and sample['video'] is not None:
            videos = sample['video']
            fps = sample.get("fps", 2.0)
            max_frames = sample.get("max_frames", 32)

            # Convert single video to list for uniform processing
            if not isinstance(videos, list):
                videos = [videos]

            # Process video files (mp4, etc.)
            for vid in videos:
                content.append({"type": "video", "video": vid, "fps": fps, "max_frames": max_frames})

        # Add text question
        content.append({"type": "text", "text": question})
        
        messages = [
            {
                "role": "user",
                "content": content,
            },
        ]
        
        return messages
    
    def prepare_vllm_input(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare vLLM input format based on backbone type
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dictionary containing prompt and multimodal data
        """
        # Handle different backbones
        if self.backbone == 'qwen3':
            # Qwen3-VL processing
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True
            )
            
            # Build multimodal data
            mm_data = {}
            if image_inputs is not None:
                mm_data['image'] = image_inputs
            if video_inputs is not None:
                mm_data['video'] = video_inputs
            
            return {
                'prompt': text,
                'multi_modal_data': mm_data,
                'mm_processor_kwargs': video_kwargs
            }
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
    
    def batch_inference(
        self,
        prepared_dataset: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.05,
        presence_penalty: float = 0.0,
        n: int = 1,
    ) -> List[List[str]]:
        """
        Batch inference with automatic vLLM optimization

        Args:
            prepared_dataset: Preprocessed dataset
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            presence_penalty: Presence penalty
            n: Number of candidates to generate per sample (for pass@k)

        Returns:
            List of lists of generated texts (each sample has n candidates)
        """
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stop_token_ids=[],
            n=n,
        )

        logger.info(f"Sampling parameters:")
        logger.info(f"  max_tokens={max_tokens}, n={n}")
        logger.info(f"  temperature={temperature}, top_p={top_p}, top_k={top_k}")
        logger.info(f"  repetition_penalty={repetition_penalty}, presence_penalty={presence_penalty}")
        
        # Prepare all inputs
        logger.info(f"\nPreparing {len(prepared_dataset)} inputs...")
        all_inputs = []
        failed_indices = []

        for idx, sample in enumerate(tqdm(prepared_dataset, desc="Preparing inputs")):
            try:
                messages = self.prepare_messages(sample)
                vllm_input = self.prepare_vllm_input(messages)
                all_inputs.append(vllm_input)
            except (ValueError, Exception) as e:
                logger.warning(f"Failed to process sample {idx}: {str(e)}")
                failed_indices.append(idx)
                # Append None as placeholder to maintain index alignment
                all_inputs.append(None)
        
        # Filter out failed samples for inference
        valid_inputs = [inp for inp in all_inputs if inp is not None]
        logger.info(f"Valid samples: {len(valid_inputs)}/{len(all_inputs)}")
        if failed_indices:
            logger.warning(f"Failed samples: {len(failed_indices)} (indices: {failed_indices}{'' if len(failed_indices) > 10 else ''})")

        # Batch inference
        logger.info(f"Starting batch inference ({len(valid_inputs)} samples)...")
        start_time = time.time()

        outputs = self.llm.generate(valid_inputs, sampling_params=sampling_params)

        elapsed = time.time() - start_time
        logger.info(f"✓ Inference completed in {elapsed:.2f}s")
        logger.info(f"  Average speed: {elapsed/len(valid_inputs):.2f}s/sample")
        logger.info(f"  Throughput: {len(valid_inputs)/elapsed:.2f} samples/s\n")

        # Extract generated texts (all n candidates for each sample)
        valid_results = []
        for output in outputs:
            candidates = [out.text for out in output.outputs]
            valid_results.append(candidates)

        # Reconstruct full results with empty predictions for failed samples
        results = []
        valid_idx = 0
        for idx in range(len(all_inputs)):
            if idx in failed_indices:
                # Empty prediction for failed samples
                results.append([])
            else:
                results.append(valid_results[valid_idx])
                valid_idx += 1

        return results


def load_and_prepare_dataset(json_path: str, base_image_dir: str) -> List[Dict[str, Any]]:
    """
    Load JSON dataset and prepare it for inference

    Args:
        json_path: Path to the JSON file
        base_image_dir: Base directory for resolving image paths

    Returns:
        List of prepared samples with 'question' and 'image' fields
    """
    logger.info(f"Loading dataset from: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[:5000]

    logger.info(f"Loaded {len(data)} samples")

    # Transform data format
    prepared_data = []
    for item in data:
        # Start with a copy of all original fields
        sample = item.copy()

        sample['question'] = item['problem']

        # Resolve image paths
        if 'images' in item and item['images']:
            image_paths = []
            for img_path in item['images']:
                full_path = os.path.join(base_image_dir, img_path)
                image_paths.append(full_path)

            # If single image, use string; if multiple, use list
            sample['image'] = image_paths[0] if len(image_paths) == 1 else image_paths

        # Resolve video paths (mp4 files only)
        if 'videos' in item and item['videos']:
            videos = item['videos']
            video_paths = []
            for vid_path in videos:
                full_path = os.path.join(base_image_dir, vid_path)
                video_paths.append(full_path)

            # If single video, use string; if multiple, use list
            sample['video'] = video_paths[0] if len(video_paths) == 1 else video_paths

        prepared_data.append(sample)

    return prepared_data


def save_results(
    original_data: List[Dict[str, Any]],
    predictions: List[List[str]],
    output_path: str
):
    """
    Save inference results to JSON file

    Args:
        original_data: Original dataset
        predictions: Model predictions (list of lists for pass@k)
        output_path: Path to save results
    """
    logger.info(f"Saving results to: {output_path}")

    results = []
    for item, pred_list in zip(original_data, predictions):
        result = item.copy()
        result['predict'] = pred_list
        results.append(result)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"✓ Results saved successfully ({len(results)} samples, {len(predictions[0])} candidates each)")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Inference parameters
    TENSOR_PARALLEL_SIZE = 8
    MAX_TOKENS = 2048
    K = 1  # Number of candidates per sample (pass@k)

    # Configuration
    model_name = "Embodied-R1.5"
    model_path = "/path/to/Embodied-R1.5"             # edit: local model path or HuggingFace ID
    base_image_dir = "/path/to/data"                  # edit: root directory for images/videos

    json_path = "rft_test_datasets/erqa.json"         # edit: input dataset JSON
    output_path = "./erqa_results.json"               # edit: where to save predictions

    logger.info("="*80)
    logger.info("Starting batch inference pipeline")
    logger.info("="*80)

    # Load original data (for saving with predictions later)
    with open(json_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    # Load and prepare dataset
    prepared_dataset = load_and_prepare_dataset(json_path, base_image_dir)

    # Initialize inference engine
    engine = VLLMInferenceEngine(
        model_path=model_path,
        model_name=model_name,
        backbone="qwen3",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=0.85,
        max_model_len=20240,
        limit_mm_per_prompt={"image": 16, "video": 1},
    )

    # Run batch inference
    predictions = engine.batch_inference(
        prepared_dataset=prepared_dataset,
        max_tokens=MAX_TOKENS,
        n=K
    )

    # Save results
    save_results(original_data, predictions, output_path)

    logger.info("="*80)
    logger.info("Pipeline completed successfully!")
    logger.info("="*80)
