"""
Unified multi-task reward function for Embodied-R1.5 data format

Supports multiple problem types:
- multiple choice: Exact match using grade_answer, reward 0/1
- numerical: Numerical comparison with 1 decimal places, reward 0/1
- open-ended: LLM Reward [0,1] / ROUGE score evaluation, reward [0,1]
- math: Symbolic equivalence verification using math_verify, reward 0/1
- spatial grounding: 2D Box IoU calculation, reward [0,1]
- trace: 2D trajectory tracking with distance-based reward, reward [0,1]
- trace_3d: 3D trajectory tracking with depth, reward [0,1]
- point: Point localization with distance-based reward, reward [0,1]

### response format example

Format: Think content (without tags) followed by <answer>...</answer>

1.multiple choice
Let me think about it.<answer>A</answer>
Let me think about it.<answer>B.dog</answer>
2.numerical
Calculating the result.<answer>42.3</answer>
3.open-ended
I think this is the answer.<answer>place sticky notes in the stand</answer>
4.math
Let me verify the equation.<answer>x = (-b ± √(b²-4ac)) / (2a)</answer>
5.spatial grounding
Locating the box.<answer>{"boxes": [100, 150, 200, 250]}</answer>
Locating the box.<answer>[{"boxes": [100, 150, 200, 250]}]</answer>
Locating the box.<answer>```json\n[{"boxes": [100, 150, 200, 250]}]\n```</answer>
6.trace
Tracking the trajectory.<answer>```json\n[{\"point_2d\": [440, 782]}, {\"point_2d\": [497, 848]}, {\"point_2d\": [567, 877]}, {\"point_2d\": [627, 880]}]\n```</answer>
7.trace_3d
Tracking the 3D trajectory.<answer>```json\n[{\"point_2d\": [440, 782], "depth": 1.3}, {\"point_2d\": [497, 848], "depth": 1.3}, {\"point_2d\": [567, 877], "depth": 1.3}, {\"point_2d\": [627, 880], "depth": 1.3}]\n```</answer>
8. point
Locating the points.<answer>```json\n[{\"point_2d\": [670, 476]}]\n```</answer>
"""
import json
import os
import random
import re
import time
from time import sleep
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import sacrebleu
from math_verify import parse as math_parse
from math_verify import verify as math_verify
from mathruler.grader import grade_answer
from rouge_score import rouge_scorer
from scipy.interpolate import interp1d
from transformers import AutoTokenizer

REWARD_NAME = "Embodied-R1.5"
REWARD_TYPE = "batch"

# Model reward for open-ended tasks
MODEL_PATH = "Skywork/Skywork-Reward-V2-Qwen3-8B"
MODEL_IP = "your ip"
MAX_RM_BATCH_SIZE = 128
USE_MODEL_FOR_OPEN_ENDED = False

# Valid values for validation
VALID_DATA_TYPES = {"image", "video", "mixed", "text"}
VALID_PROBLEM_TYPES = {
    "multiple choice", "trace", "open-ended", "math",
    "numerical", "point", "spatial grounding", "trace_3d"
}
REQUIRED_KEYS = {"response", "response_length", "ground_truth", "data_type", "problem_type", "problem_id", "problem"}

# -------------------------
# Patterns for format check
# -------------------------
# Only check for <answer> tags, think content can be anywhere before it
ANSWER_CAPTURE_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.DOTALL
)


# ===================== Wrapper: batch call external model for open-ended =====================
class RewardModelClient:
    """Reward client for Skywork reward model using sglang server."""

    def __init__(self, model_path, model_ip, port=18889, server_num=1):
        """
        Args:
            model_path: Path to the Skywork reward model.
            base_urls: List of server URLs for the sglang /classify endpoint.
        """
        self.model_path = model_path
        self.base_urls = [f"http://{model_ip}:{port + i}/classify" for i in range(server_num)]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.current_url_idx = 0

    def _get_next_url(self):
        """Get next URL for load balancing."""
        url = self.base_urls[self.current_url_idx]
        self.current_url_idx = (self.current_url_idx + 1) % len(self.base_urls)
        return url

    def __call__(self, convs, base_url=None, retry_delay=0.2, max_retries=5, timeout=20):
        """Process conversations and return reward scores.

        Args:
            convs: List of conversations, where each conversation is a list of messages
                   with 'role' and 'content' keys.
            base_url: Optional specific server URL. If None, uses load balancing.
            retry_delay: Delay in seconds before retrying the request.
            max_retries: Maximum number of retries for the request.
            timeout: Request timeout in seconds.

        Returns:
            List of reward scores, or list of None values if error occurs.
        """
        if base_url is None:
            base_url = self._get_next_url()

        payload = {"model": self.model_path}
        convs_formatted = []
        for conv in convs:
            conv_str = self.tokenizer.apply_chat_template(conv, tokenize=False)
            if self.tokenizer.bos_token is not None and conv_str.startswith(self.tokenizer.bos_token):
                conv_str = conv_str[len(self.tokenizer.bos_token):]
            convs_formatted.append(conv_str)

        payload.update({"text": convs_formatted})

        # Retry logic
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    base_url,
                    json=payload,
                    proxies={"http": None, "https": None},  # Disable proxy
                    timeout=timeout
                )
                response.raise_for_status()
                rewards = [item["embedding"][0] for item in response.json()]
                assert len(rewards) == len(convs), f"Expected {len(convs)} rewards, got {len(rewards)}"
                return rewards
            except Exception as e:
                print(f"Error requesting reward (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    sleep(retry_delay)
                else:
                    print(f"Failed to request reward after {max_retries} retries")
                    return [None] * len(convs)


def evaluate_open_ended_with_rm(
    open_ended_queue: List[Dict[str, Any]],
    results: List[Dict[str, float]],
    format_weight: float,
    rm_batch_size: int,
    temperature: float = 1.0
) -> None:
    """
    Take open-ended samples in open_ended_queue, and call external RM in batches to evaluate accuracy.
    Failed batches fall back to ROUGE. Optionally apply mean-std → min-max normalization within
    each problem_id group.
    After evaluation, this function will fill results[idx]['accuracy'] in-place and recompute
    results[idx]['overall'].
    """
    # print(USE_MODEL_FOR_OPEN_ENDED)
    # print(open_ended_queue)
    if not USE_MODEL_FOR_OPEN_ENDED or not open_ended_queue:
        return

    client = RewardModelClient(
        MODEL_PATH,
        MODEL_IP,
        port=18889,
        server_num=1
    )

    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    model_scores: List[float] = [0.0] * len(open_ended_queue)

    # Build conversations for batch processing
    for batch_id, batch in enumerate(_chunks(open_ended_queue, rm_batch_size)):
        # Format conversations for the reward model
        batch_convs = []
        for b in batch:
            prompt = b["prompt"]
            reference = b["reference"]
            # Extract only the answer part, not the thinking process
            output = extract_answer(b["output"])

            # Construct prompt with explicit similarity evaluation instruction
            prompt_with_ref = (
                f"{prompt}\n\n"
                f"Reference answer: {reference}\n\n"
                f"Compare the assistant's response with the reference answer above. "
                f"Score based on how closely the response matches the reference in meaning and content."
            )

            conv = [
                {"role": "user", "content": prompt_with_ref},
                {"role": "assistant", "content": output}
            ]
            batch_convs.append(conv)

        try:
            # Call reward model with properly formatted conversations
            rewards = client(batch_convs)  # expected to return list[float]
            # print(rewards)

            # Check if any rewards are None (indicating error)
            if rewards is None or any(r is None for r in rewards):
                raise Exception("Reward model returned None values")

            # Store the scores with temperature-adjusted sigmoid normalization
            # Temperature (default 1.0) for score range [-5, 5] to avoid saturation
            # Maps unbounded scores to [0, 1] range with better discrimination
            for j, sc in enumerate(rewards):
                # Sigmoid with temperature: 1 / (1 + exp(-x/T))
                normalized_score = 1.0 / (1.0 + np.exp(-float(sc) / temperature))
                model_scores[(batch_id * rm_batch_size) + j] = normalized_score
            # print(model_scores)

        except Exception as e:
            print(f"Batch {batch_id} failed with error: {e}. Falling back to BLEU.")
            # Fallback: use BLEU to compute scores for this batch
            for j, b in enumerate(batch):
                ref = b["reference"]
                output = extract_answer(b["output"])
                bleu_score = compute_bleu_score(ref, output)
                model_scores[(batch_id * rm_batch_size) + j] = float(max(0.0, min(1.0, bleu_score)))

    # Per-problem_id normalization removed to avoid reward hacking
    # Now using only sigmoid normalization from lines 209-214

    # Fill back accuracy, and recompute overall
    for k, b in enumerate(open_ended_queue):
        idx = b["idx"]
        results[idx]["accuracy"] = float(max(0.0, min(1.0, model_scores[k])))
        results[idx]["overall"] = (
            (1.0 - format_weight) * results[idx]["accuracy"]
            + format_weight * results[idx]["format"]
        )
# ==================================================================

# -------------------------
# Helper functions
# -------------------------
def _json(s):
    """Parse JSON from string, handling markdown code blocks and escape sequences"""
    try:
        # Remove markdown code block markers
        text = re.sub(r'```json\s*', '', s)
        text = re.sub(r'```', '', text)
        # Also remove ''' (common typo for ```)
        text = text.replace("'''", "")
        text = text.strip()

        # Handle escaped newlines and quotes that might appear in model output
        # Replace literal \n with actual newlines, then strip them
        text = text.replace('\\n', '\n').replace('\n', '')
        # Replace literal \" with actual quotes
        text = text.replace('\\"', '"')

        return json.loads(text)
    except Exception:
        return None


def _is_list_of_numbers(obj, expected_len: int) -> bool:
    """Check if obj is a list of numbers with expected length"""
    return (
        isinstance(obj, list)
        and len(obj) == expected_len
        and all(isinstance(x, (int, float)) for x in obj)
    )


def extract_answer(text: str) -> str:
    """Extract content from <answer> tags"""
    match = ANSWER_CAPTURE_PATTERN.search(text or "")
    return match.group(1).strip() if match else ""


def normalize_number(num_str: str) -> Optional[float]:
    try:
        return float((num_str or "").replace(",", ""))
    except Exception:
        return None


def iou_2d(box1: List[float], box2: List[float]) -> float:
    # Strict: must be numeric lists with length 4; otherwise return 0
    if not _is_list_of_numbers(box1, 4) or not _is_list_of_numbers(box2, 4):
        return 0.0
    try:
        x1, y1, x2, y2 = map(float, box1)
        X1, Y1, X2, Y2 = map(float, box2)
    except Exception:
        return 0.0
    inter_x1, inter_y1 = max(x1, X1), max(y1, Y1)
    inter_x2, inter_y2 = min(x2, X2), min(y2, Y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, X2 - X1) * max(0.0, Y2 - Y1)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 1e-12 else 0.0


def grade_multiple_choice(ans: str, gt: str) -> bool:
    """
    Grade multiple choice answers with flexible matching.

    Handles cases where:
    - gt is "A" and ans is "A.dog" -> should be correct
    - gt is "A" and ans is "A" -> should be correct
    - gt is the full answer text and ans matches it -> should be correct
    - Ordering/ranking questions (e.g., "1-3-2") -> strict exact match

    Args:
        ans: Model's answer (e.g., "A", "A.dog", "dog", "1-3-2")
        gt: Ground truth (e.g., "A", "B", "dog", "1-3-2")

    Returns:
        True if answer is correct, False otherwise
    """
    ans_stripped = ans.strip()
    gt_stripped = gt.strip()

    # Special handling for ordering/ranking questions
    # If ground truth contains digits and hyphens/commas (e.g., "1-3-2", "1,3,2"), use strict matching
    if re.search(r'\d+[-,]\d+', gt_stripped):
        return ans_stripped == gt_stripped

    # First try exact match using grade_answer
    if grade_answer(ans_stripped, gt_stripped):
        return True

    # Check if gt is a single letter option (A, B, C, D, etc.)
    if len(gt_stripped) == 1 and gt_stripped.isalpha():
        # Check if answer starts with the correct option letter
        # Handle formats like "A", "A.", "A:", "A)", "A.dog", "A: dog", etc.
        if ans_stripped.upper().startswith(gt_stripped.upper()):
            # Make sure it's actually the option letter, not just coincidentally starting with that letter
            # Check if it's followed by a separator or is exactly the letter
            if len(ans_stripped) == 1:
                return True
            # Check if followed by common separators
            next_char = ans_stripped[1]
            if next_char in '.,:;) \t':
                return True

    return False


def math_equivalent(gt: str, pred: str) -> bool:
    """
    Use math_verify to perform symbolic equivalence checking; if it fails (exceptions, etc.),
    fall back to grade_answer.
    """
    try:
        return bool(math_verify(math_parse(gt), math_parse(pred)))
    except Exception:
        return grade_answer(pred, gt)


def compute_rouge_score(reference: str, hypothesis: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference or "", hypothesis or "")
    return (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3.0


def compute_bleu_score(reference: str, hypothesis: str) -> float:
    """
    Compute BLEU score as the average of BLEU-1, BLEU-2, and BLEU-3.
    Optimized for short sentences (< 10 words).

    Args:
        reference: Reference text (ground truth)
        hypothesis: Hypothesis text (model prediction)

    Returns:
        Average of BLEU-1, BLEU-2, and BLEU-3 scores, normalized to [0, 1]
    """
    try:
        # Compute BLEU-1 (unigram matches)
        bleu1 = sacrebleu.sentence_bleu(hypothesis or "", [reference or ""], tokenize="intl", max_ngram_order=1, smooth_method='exp').score

        # Compute BLEU-2 (bigram matches)
        bleu2 = sacrebleu.sentence_bleu(hypothesis or "", [reference or ""], tokenize="intl", max_ngram_order=2, smooth_method='exp').score

        # Compute BLEU-3 (trigram matches)
        bleu3 = sacrebleu.sentence_bleu(hypothesis or "", [reference or ""], tokenize="intl", max_ngram_order=3, smooth_method='exp').score

        # Return simple average, normalized to [0, 1] (BLEU scores are in range 0-100)
        return (bleu1 + bleu2 + bleu3) / 3.0 / 100.0
    except Exception:
        return 0.0


def interpolate_trajectory(trajectory: List[List[float]], new_length: int) -> List[List[float]]:
    """
    Interpolate trajectory to a fixed number of points using linear interpolation

    Args:
        trajectory: List of [x, y] coordinates
        new_length: Target number of points

    Returns:
        Interpolated trajectory with new_length points
    """
    if new_length <= 1:
        return trajectory

    # Handle single point case: replicate the point to all positions
    # This prevents reward hacking where model outputs only 1 point
    if len(trajectory) == 1:
        return [trajectory[0][:] for _ in range(new_length)]

    old_indices = np.arange(len(trajectory))
    new_indices = np.linspace(0, len(trajectory) - 1, new_length)

    x_coords = [p[0] for p in trajectory]
    y_coords = [p[1] for p in trajectory]

    x_interpolator = interp1d(old_indices, x_coords, kind='linear')
    y_interpolator = interp1d(old_indices, y_coords, kind='linear')

    new_x_coords = x_interpolator(new_indices)
    new_y_coords = y_interpolator(new_indices)

    return [[float(x), float(y)] for x, y in zip(new_x_coords, new_y_coords)]


def accuracy_reward_trace(
    response: str,
    ground_truth: str,
    perfect_threshold: float = 50.0,
    zero_threshold: float = 110.0,
    length_mismatch_penalty: float = 0.35
) -> float:
    """
    Trajectory tracking accuracy for 2D point sequences
    Calculates RMSE distance between predicted and ground truth trajectories

    Args:
        response: Model's predicted trajectory (JSON array of points)
        ground_truth: Expected trajectory (JSON array of points)
        perfect_threshold: RMSE below this value gets reward 1.0 (default: 50.0 pixels)
        zero_threshold: RMSE above this value gets reward 0.0 (default: 110.0 pixels)
        length_mismatch_penalty: Penalty applied when point counts don't match (default: 0.5)

    Returns:
        Reward in [0, 1] based on RMSE distance
        - If point counts don't match: apply length_mismatch_penalty, then interpolate both trajectories to max length for RMSE calculation
        - RMSE < perfect_threshold: reward = 1.0
        - perfect_threshold ≤ RMSE < zero_threshold: linear decay from 1.0 to 0.0
        - RMSE ≥ zero_threshold: reward = 0.0

    Note:
        Distance thresholds may need adjustment based on image resolution and task requirements
        Interpolation ensures fair comparison when trajectory lengths differ
    """
    try:
        pred_points = _json(response)
        gt_points = _json(ground_truth)

        if pred_points is None or gt_points is None:
            return 0.0

        # Check if number of points match
        length_penalty = 0.0
        if len(pred_points) != len(gt_points):
            length_penalty = length_mismatch_penalty
            # print("length_mismatch_penalty")

        # Extract point_2d coordinates
        pred_trajectory = [p.get("point_2d", [0, 0]) for p in pred_points]
        gt_trajectory = [p.get("point_2d", [0, 0]) for p in gt_points]

        if len(pred_trajectory) == 0 or len(gt_trajectory) == 0:
            return 0.0

        # Trajectory must have at least 2 points - single point is invalid
        if len(pred_trajectory) == 1:
            return 0.0

        # Interpolate to max length if lengths don't match
        if len(pred_trajectory) != len(gt_trajectory):
            max_length = max(len(pred_trajectory), len(gt_trajectory))
            pred_trajectory = interpolate_trajectory(pred_trajectory, max_length)
            gt_trajectory = interpolate_trajectory(gt_trajectory, max_length)
        # print(pred_trajectory, gt_trajectory)
        # Calculate squared distances on aligned trajectories
        squared_distances = 0.0
        for pred_xy, gt_xy in zip(pred_trajectory, gt_trajectory):
            squared_distance = (pred_xy[0] - gt_xy[0])**2 + (pred_xy[1] - gt_xy[1])**2
            squared_distances += squared_distance

        # Calculate RMSE (Root Mean Squared Error)
        num_points = len(pred_trajectory)
        rmse = (squared_distances / num_points)**0.5
        # print(rmse)

        # Apply distance thresholds
        if rmse < perfect_threshold:
            base_reward = 1.0
        elif rmse < zero_threshold:
            # Linear decay between perfect_threshold and zero_threshold
            decay_range = zero_threshold - perfect_threshold
            base_reward = max(0.0, 1.0 - (rmse - perfect_threshold) / decay_range)
        else:
            base_reward = 0.0

        # Apply length penalty
        final_reward = max(0.0, base_reward - length_penalty)

        return final_reward

    except Exception:
        return 0.0


def accuracy_reward_trace_3d(
    response: str,
    ground_truth: str,
    perfect_threshold_2d: float = 50.0,
    zero_threshold_2d: float = 110.0,
    perfect_threshold_depth: float = 0.05,
    zero_threshold_depth: float = 0.3,
    length_mismatch_penalty: float = 0.35,
    weight_2d: float = 0.5,
    weight_depth: float = 0.5
) -> float:
    """
    3D trajectory tracking accuracy with separate 2D and depth evaluation
    Calculates weighted combination of 2D trajectory RMSE and depth average absolute difference

    Args:
        response: Model's predicted 3D trajectory (JSON array of points with depth)
        ground_truth: Expected 3D trajectory (JSON array of points with depth)
        perfect_threshold_2d: 2D RMSE below this gets reward 1.0 (default: 8.0 pixels)
        zero_threshold_2d: 2D RMSE above this gets reward 0.0 (default: 50.0 pixels)
        perfect_threshold_depth: Depth MAE below this gets reward 1.0 (default: 0.05 meters)
        zero_threshold_depth: Depth MAE above this gets reward 0.0 (default: 0.3 meters)
        length_mismatch_penalty: Penalty when point counts don't match (default: 0.5)
        weight_2d: Weight for 2D trajectory reward (default: 0.5)
        weight_depth: Weight for depth reward (default: 0.5)

    Returns:
        Reward in [0, 1] as weighted combination: weight_2d * reward_2d + weight_depth * reward_depth
        - If point counts don't match: apply penalty, then interpolate to max length
        - 2D component: RMSE-based reward on x,y coordinates (similar to accuracy_reward_trace)
        - Depth component: MAE-based reward on depth values (average absolute difference)

    Note:
        weight_2d + weight_depth should equal 1.0 for proper normalization
        Interpolation ensures fair comparison when trajectory lengths differ
    """
    try:
        pred_points = _json(response)
        gt_points = _json(ground_truth)

        if pred_points is None or gt_points is None:
            return 0.0

        # Handle both single object and list of objects
        if isinstance(pred_points, dict):
            pred_points = [pred_points]
        if isinstance(gt_points, dict):
            gt_points = [gt_points]

        # Check if number of points match
        length_penalty = 0.0
        if len(pred_points) != len(gt_points):
            length_penalty = length_mismatch_penalty

        # Extract 2D coordinates and depth values separately
        pred_trajectory_2d = [p.get("point_2d", [0, 0]) for p in pred_points]
        gt_trajectory_2d = [p.get("point_2d", [0, 0]) for p in gt_points]
        pred_depths = [p.get("depth", 0.0) for p in pred_points]
        gt_depths = [p.get("depth", 0.0) for p in gt_points]

        if len(pred_trajectory_2d) == 0 or len(gt_trajectory_2d) == 0:
            return 0.0

        # Trajectory must have at least 2 points - single point is invalid
        if len(pred_trajectory_2d) == 1:
            return 0.0

        # Interpolate to max length if lengths don't match
        if len(pred_trajectory_2d) != len(gt_trajectory_2d):
            max_length = max(len(pred_trajectory_2d), len(gt_trajectory_2d))
            pred_trajectory_2d = interpolate_trajectory(pred_trajectory_2d, max_length)
            gt_trajectory_2d = interpolate_trajectory(gt_trajectory_2d, max_length)

            # Interpolate depth values using numpy
            pred_depths = np.interp(
                np.linspace(0, len(pred_depths) - 1, max_length),
                np.arange(len(pred_depths)),
                pred_depths
            ).tolist()
            gt_depths = np.interp(
                np.linspace(0, len(gt_depths) - 1, max_length),
                np.arange(len(gt_depths)),
                gt_depths
            ).tolist()

        # Calculate 2D RMSE
        squared_distances_2d = 0.0
        for pred_xy, gt_xy in zip(pred_trajectory_2d, gt_trajectory_2d):
            squared_distance = (pred_xy[0] - gt_xy[0])**2 + (pred_xy[1] - gt_xy[1])**2
            squared_distances_2d += squared_distance

        num_points = len(pred_trajectory_2d)
        rmse_2d = (squared_distances_2d / num_points)**0.5

        # Convert 2D RMSE to reward
        if rmse_2d < perfect_threshold_2d:
            reward_2d = 1.0
        elif rmse_2d < zero_threshold_2d:
            decay_range = zero_threshold_2d - perfect_threshold_2d
            reward_2d = max(0.0, 1.0 - (rmse_2d - perfect_threshold_2d) / decay_range)
        else:
            reward_2d = 0.0

        # Calculate depth MAE (Mean Absolute Error)
        absolute_errors_depth = sum(abs(pd - gd) for pd, gd in zip(pred_depths, gt_depths))
        mae_depth = absolute_errors_depth / num_points

        # Convert depth MAE to reward
        if mae_depth < perfect_threshold_depth:
            reward_depth = 1.0
        elif mae_depth < zero_threshold_depth:
            decay_range = zero_threshold_depth - perfect_threshold_depth
            reward_depth = max(0.0, 1.0 - (mae_depth - perfect_threshold_depth) / decay_range)
        else:
            reward_depth = 0.0

        # Combine rewards with weights
        base_reward = weight_2d * reward_2d + weight_depth * reward_depth

        # Apply length penalty
        final_reward = max(0.0, base_reward - length_penalty)

        return final_reward

    except Exception:
        return 0.0


def point_in_polygon(point: List[float], polygon: List[List[float]]) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def point_in_box(point: List[float], box: List[float]) -> bool:
    """Check if a point is inside a bounding box [x1, y1, x2, y2]"""
    if len(box) != 4 or len(point) != 2:
        return False
    x, y = point[0], point[1]
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def find_nearest_match(pred_points: List[List[float]], gt_points: List[List[float]]) -> float:
    """
    Find nearest match between predicted and ground truth points.
    Returns the average minimum distance.
    """
    if not pred_points or not gt_points:
        return float('inf')

    total_distance = 0.0
    for pred_pt in pred_points:
        min_dist = float('inf')
        for gt_pt in gt_points:
            dist = ((pred_pt[0] - gt_pt[0])**2 + (pred_pt[1] - gt_pt[1])**2)**0.5
            min_dist = min(min_dist, dist)
        total_distance += min_dist

    return total_distance / len(pred_points)


def accuracy_reward_point(
    response: str,
    ground_truth: str,
    perfect_threshold: float = 15.0,
    zero_threshold: float = 100.0,
    count_penalty: float = 0.3,
) -> float:
    """
    Point localization accuracy with support for multiple formats:
    1. With count: Check count match, then nearest distance matching
    2. Pure point_2d: Calculate minimum distance sum
    3. Segmentation (polygon): Check if points are inside polygon
    4. Box_2d: Check if points are inside box

    Supports partial correctness: if 2 out of 3 points are correct, score is 0.67

    Args:
        response: Model's predicted point(s) (JSON object or array)
        ground_truth: Expected point(s) (JSON object or array)
        perfect_threshold: Distance below this gets reward 1.0 (default: 15.0 pixels)
        zero_threshold: Distance above this gets reward 0.0 (default: 40.0 pixels)

    Returns:
        Reward in [0, 1] based on correctness
    """
    try:
        ans = response
        gt = ground_truth or ""

        pred_data = _json(ans)
        gt_data = _json(gt)

        if pred_data is None or gt_data is None:
            return 0.0

        if isinstance(pred_data, dict):
            pred_data = [pred_data]
        if isinstance(gt_data, dict):
            gt_data = [gt_data]

        pred_points = []
        for item in pred_data:
            if "point_2d" in item:
                pred_points.append(item["point_2d"])

        if not pred_points:
            return 0.0

        # Case 1: Ground truth has segmentation (polygon)
        gt_segmentations = []
        for item in gt_data:
            if "segmentation" in item:
                gt_segmentations.append(item["segmentation"])

        if gt_segmentations:
            # Check how many predicted points are inside any of the polygons
            correct_count = 0
            for pt in pred_points:
                # Point is correct if it's inside ANY of the polygons
                if any(point_in_polygon(pt, seg) for seg in gt_segmentations):
                    correct_count += 1
            return correct_count / len(pred_points) if pred_points else 0.0

        # Case 2: Ground truth has box_2d
        gt_box = None
        for item in gt_data:
            if "box_2d" in item:
                gt_box = item["box_2d"]
                break

        if gt_box is not None:
            # Check how many predicted points are inside the box
            correct_count = sum(1 for pt in pred_points if point_in_box(pt, gt_box))
            return correct_count / len(pred_points) if pred_points else 0.0

        # Case 3: Point_2d matching with optional count penalty
        # Extract ground truth points
        gt_points = []
        for item in gt_data:
            if "point_2d" in item:
                gt_points.append(item["point_2d"])

        if not gt_points:
            return 0.0

        # Check if ground truth has count field
        gt_count = None
        for item in gt_data:
            if "count" in item:
                gt_count = item["count"]
                break

        # Apply count penalty if count field exists and doesn't match
        penalty = 0.0
        if gt_count is not None and len(pred_points) != gt_count:
            penalty = count_penalty
            # print("count_penalty!!!")

        # Calculate average minimum distance
        avg_dist = find_nearest_match(pred_points, gt_points)
        # print(avg_dist)

        # Distance-based reward with configurable thresholds
        if avg_dist < perfect_threshold:
            base_reward = 1.0
        elif avg_dist < zero_threshold:
            decay_range = zero_threshold - perfect_threshold
            base_reward = max(0.0, 1.0 - (avg_dist - perfect_threshold) / decay_range)
        else:
            base_reward = 0.0

        # Apply count penalty if applicable
        return max(0.0, base_reward - penalty)

    except Exception:
        return 0.0


def format_structure_reward_check(response: str, problem_type: str) -> float:
    """
    Combined format and structure check.

    New format: Think content (without tags) followed by <answer>...</answer>

    Checks both:
    1. Format: <answer>...</answer> structure (think content can be anywhere before it)
    2. Structure: Task-specific JSON structure requirements

    Args:
        response: Full model response (think content + <answer>...</answer>)
        problem_type: Type of problem

    Returns:
        1.0 if both format and structure are correct, 0.0 otherwise
    """
    # Extract answer content
    answer = extract_answer(response)
    if not answer:
        return 0.0

    # Then check structure based on problem type
    ptype = (problem_type or "").lower()

    # For point: {"point_2d": [x, y]} or [{"point_2d": [x, y]}, ...]
    if ptype == "point":
        obj = _json(answer)
        if isinstance(obj, dict):
            ok = _is_list_of_numbers(obj.get("point_2d"), 2)
        elif isinstance(obj, list):
            ok = len(obj) > 0 and all(isinstance(item, dict) and _is_list_of_numbers(item.get("point_2d"), 2) for item in obj)
        else:
            ok = False
        return 1.0 if ok else 0.0

    # For trace: {"point_2d": [x, y]} or [{"point_2d": [x, y]}, ...]
    if ptype == "trace":
        obj = _json(answer)
        if isinstance(obj, dict):
            ok = _is_list_of_numbers(obj.get("point_2d"), 2)
        elif isinstance(obj, list):
            ok = len(obj) > 0 and all(isinstance(item, dict) and _is_list_of_numbers(item.get("point_2d"), 2) for item in obj)
        else:
            ok = False
        return 1.0 if ok else 0.0

    # For trace_3d: {"point_2d": [x, y], "depth": float} or [{"point_2d": [x, y], "depth": float}, ...]
    if ptype == "trace_3d":
        obj = _json(answer)
        if isinstance(obj, dict):
            ok = _is_list_of_numbers(obj.get("point_2d"), 2) and isinstance(obj.get("depth"), (int, float))
        elif isinstance(obj, list):
            ok = (
                len(obj) > 0
                and all(
                    isinstance(item, dict)
                    and _is_list_of_numbers(item.get("point_2d"), 2)
                    and isinstance(item.get("depth"), (int, float))
                    for item in obj
                )
            )
        else:
            ok = False
        return 1.0 if ok else 0.0

    # For spatial grounding: {"boxes": [x1, y1, x2, y2]} or [{"boxes": [x1, y1, x2, y2]}, ...]
    if ptype == "spatial grounding":
        obj = _json(answer)
        if isinstance(obj, dict):
            ok = _is_list_of_numbers(obj.get("boxes"), 4)
        elif isinstance(obj, list):
            ok = len(obj) > 0 and all(isinstance(item, dict) and _is_list_of_numbers(item.get("boxes"), 4) for item in obj)
        else:
            ok = False
        return 1.0 if ok else 0.0

    # For multiple choice, open-ended, math, numerical: format check is sufficient
    return 1.0


# ------------------------------------------
# Accuracy reward (normalized to [0,1] for all)
# ------------------------------------------
def accuracy_reward(response: str,
                    ground_truth: str,
                    problem_type: str) -> float:
    """
    Normalized accuracy ∈ [0,1]. Strict format requirement: if the format is invalid, always return 0.
    Wrapped with try/except: any exception → 0.0.
    """
    try:
        ans = extract_answer(response)
        ptype = (problem_type or "").lower()
        gt = ground_truth or ""

        # ------ Pure QA type ------
        if ptype == "multiple choice":
            # answer: A | A.dog | dog | A:dog | A) dog
            return 1.0 if grade_multiple_choice(ans.strip(), gt.strip()) else 0.0

        if ptype == "numerical":
            # answer: 3.13 | 3,130.00
            gt_num, pr_num = normalize_number(gt), normalize_number(ans)
            return 1.0 if (gt_num is not None and pr_num is not None and round(gt_num, 1) == round(pr_num, 1)) else 0.0

        if ptype == "open-ended":
            # answer: free text
            # only for no RM
            # ROUGE
            # return max(0.0, min(1.0, compute_rouge_score(gt, ans)))
            # BLEU
            return max(0.0, min(1.0, compute_bleu_score(gt, ans)))

        if ptype == "math":
            # answer: mathematical expression
            return 1.0 if math_equivalent(gt, ans) else 0.0

        # spatial grounding: box IoU ∈ [0,1]
        if ptype == "spatial grounding":
            # answer: {"boxes": [x1, y1, x2, y2]} or [{"boxes": [x1, y1, x2, y2]}, ...]
            pred = _json(ans)
            gtj  = _json(gt)
            if isinstance(pred, list):
                pred = pred[0]  # only evaluate the first box
            if not isinstance(pred, dict) or not isinstance(gtj, dict):
                return 0.0
            return iou_2d(pred["boxes"], gtj["boxes"])

        # trace: trajectory distance-based reward ∈ [0,1]
        if ptype == "trace":
            return accuracy_reward_trace(ans,
                                        ground_truth,
                                        perfect_threshold=50.0,
                                        zero_threshold=120.0,
                                        length_mismatch_penalty=0.35)

        if ptype == "trace_3d":
            return accuracy_reward_trace_3d(ans,
                                            ground_truth,
                                            perfect_threshold_2d = 50.0,
                                            zero_threshold_2d = 130.0,
                                            perfect_threshold_depth = 0.1,
                                            zero_threshold_depth = 0.4,
                                            length_mismatch_penalty = 0.35,
                                            weight_2d = 0.5,
                                            weight_depth = 0.5)

        if ptype == "point":
            return accuracy_reward_point(ans, ground_truth,
                                            perfect_threshold = 40.0,
                                            zero_threshold = 150.0,
                                            count_penalty = 0.3)

        # Unknown type
        return 0.0
    except Exception:
        # Outer fallback: any exception will be scored as 0
        return 0.0


def compute_score(reward_inputs: List[Dict[str, Any]],
                  format_weight: float = 0.1) -> List[Dict[str, float]]:
    """
    New format: Think content (without tags) followed by <answer>...</answer>

    Unified multi-task reward computation entry point

    Args:
        reward_inputs: List of dicts containing response, ground_truth, problem_type, etc.
        format_weight: Weight for format score (default 0.1)

    Returns:
        List of score dicts containing overall, format, and accuracy scores

    Batch input example:
        Each item:
        {
            "response": str,
            "response_length": int,
            "ground_truth": str,   # may also contain <answer>...</answer>, here we extract it first
            "data_type": str,      # "image" | "video" | "mixed" | "text"
            "problem_type": str    # "multiple choice" | "trace" | "open-ended" | "math" | "numerical" | "point" | "spatial grounding" | "trace_3d"
            "problem_id": Any     # grouping id
            "problem": str        # used as prompt for external RM in open-ended tasks
            "dataset_name": str   # optional, for logging purposes
        }
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")

    results: List[Dict[str, float]] = []
    open_ended_queue = []

    for idx, reward_input in enumerate(reward_inputs):
        try:
            # 1. Validate
            # Validate required keys are present
            assert all(key in reward_input for key in REQUIRED_KEYS), \
                f"Missing required keys. Expected: {REQUIRED_KEYS}, Got: {set(reward_input.keys())}"

            # Validate data_type is valid
            data_type = reward_input["data_type"]
            assert data_type in VALID_DATA_TYPES, \
                f"Invalid data_type: '{data_type}'. Must be one of: {VALID_DATA_TYPES}"

            # Validate problem_type is valid
            problem_type = reward_input["problem_type"]
            assert problem_type in VALID_PROBLEM_TYPES, \
                f"Invalid problem_type: '{problem_type}'. Must be one of: {VALID_PROBLEM_TYPES}"

            # 2. Extract fields
            ground_truth = reward_input["ground_truth"]
            response = reward_input["response"]

            # 3. Format and structure check
            # Combined check for format (<answer>...</answer>) and structure
            format_structure_score = format_structure_reward_check(response, problem_type)

            # 4. Accuracy (all normalized to [0,1])
            # If format is invalid (e.g. missing <answer> tags), we still try to grade the answer.
            # To keep minimal changes and reuse existing accuracy_reward(), we wrap the full
            # response into <answer>...</answer> when no answer tag is found.
            response_for_accuracy = response
            if ANSWER_CAPTURE_PATTERN.search(response or "") is None:
                response_for_accuracy = f"<answer>{response or ''}</answer>"

            if USE_MODEL_FOR_OPEN_ENDED and problem_type.lower() == "open-ended":
                # First set to 0, and finally compute with external model and fill back
                answer_score = 0.0
                open_ended_queue.append({
                    "idx": idx,
                    "prompt": reward_input.get("problem"),
                    "reference": ground_truth or "",
                    "output": response_for_accuracy,
                    "problem_id": reward_input.get("problem_id"),
                })
            else:
                answer_score = accuracy_reward(response_for_accuracy, ground_truth, problem_type)
                # print("answer score", answer_score)

            # 5. Overall score: weighted average
            # format_structure_score = format * structure (both must be 1 to get 1)
            overall_score = (1 - format_weight) * answer_score + format_weight * format_structure_score

            results.append({
                "overall": overall_score,
                "format": format_structure_score,
                "accuracy": answer_score,
                "dataset_name": reward_input.get("dataset_name", None),
            })
        except Exception as e:
            print(f"Error computing reward for sample {idx}: {e}")
            # Fallback for the entire sample: any exception, all fields are set to 0
            results.append({
                "overall": 0.0,
                "format": 0.0,
                "accuracy": 0.0,
                "dataset_name": reward_input.get("dataset_name", None),
            })

    # ===================== Call wrapper for batch external evaluation and fill back =====================
    evaluate_open_ended_with_rm(
        open_ended_queue=open_ended_queue,
        results=results,
        format_weight=format_weight,
        rm_batch_size=MAX_RM_BATCH_SIZE
    )
    # ======================================================================

    # if random.random() < 0.01:
    #     for idx, item in enumerate(reward_inputs):
    #         print('type:', item.get("problem_type", ""))
    #         print('gt:', item.get("ground_truth", ""))
    #         print('ans:', item.get("response", ""))
    #         print({
    #             "overall": results[idx]["overall"],
    #             "format": results[idx]["format"],
    #             "accuracy": results[idx]["accuracy"],
    #             "dataset_name": results[idx].get("dataset_name", None),
    #         })
    return results
