import time
from time import sleep

import numpy as np
import requests
from transformers import AutoTokenizer

# Config for Skywork reward model
MODEL_PATH = "/path/to/Skywork-Reward-V2-Qwen3-8B"  # edit: path to Skywork reward model
MODEL_IP = "127.0.0.1"                               # edit: IP address of the sglang reward server
print("MODEL IP: ", MODEL_IP)
print("MODEL PATH:",  MODEL_PATH)

# ===================== Wrapper: batch call external model for open-ended =====================
class RewardModelClient:
    """Reward client for Skywork reward model using sglang server."""

    def __init__(self, model_path=MODEL_PATH, port=18889, server_num=1):
        """
        Args:
            model_path: Path to the Skywork reward model.
            base_urls: List of server URLs for the sglang /classify endpoint.
        """
        print("=" * 80)
        print("[DEBUG] RewardModelClient initialized!")
        print(f"[DEBUG] Model path: {model_path}")
        print(f"[DEBUG] Server IP: {MODEL_IP}")
        print(f"[DEBUG] Port: {port}, Server num: {server_num}")
        self.model_path = model_path
        self.base_urls = [f"http://{MODEL_IP}:{port + i}/classify" for i in range(server_num)]
        print(f"[DEBUG] Base URLs: {self.base_urls}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.current_url_idx = 0
        print("[DEBUG] RewardModelClient initialization complete!")
        print("=" * 80)

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

if __name__ == "__main__":
    client = RewardModelClient()

    print("\n" + "=" * 50)
    print("Reward Range Test with Diverse Examples")
    print("=" * 50)

    # Test cases with varying quality levels
    test_cases = [
        {
            "prompt": "What is 2+2?",
            "reference": "4",
            "responses": {
                "perfect": "4",
                "good": "2+2 equals 4.",
                "partial": "It's around 4 or 5.",
                "wrong": "2+2 equals 5.",
                "nonsense": "The answer is banana."
            }
        },
        {
            "prompt": "What is the capital of France?",
            "reference": "Paris",
            "responses": {
                "perfect": "Paris",
                "good": "The capital of France is Paris.",
                "partial": "I think it's Paris or Lyon.",
                "wrong": "The capital of France is London.",
                "nonsense": "France doesn't have a capital."
            }
        },
        {
            "prompt": "How many days are in a week?",
            "reference": "7 days",
            "responses": {
                "perfect": "7 days",
                "good": "There are 7 days in a week.",
                "partial": "About 7 or 8 days.",
                "wrong": "There are 5 days in a week.",
                "nonsense": "Days don't exist in weeks."
            }
        },
        {
            "prompt": "What color is the sky?",
            "reference": "Blue",
            "responses": {
                "perfect": "Blue",
                "good": "The sky is blue.",
                "partial": "It's blue or sometimes gray.",
                "wrong": "The sky is green.",
                "nonsense": "Sky has no color."
            }
        },
        {
            "prompt": "How many legs does a dog have?",
            "reference": "4 legs",
            "responses": {
                "perfect": "4 legs",
                "good": "A dog has 4 legs.",
                "partial": "Usually 4, sometimes 3.",
                "wrong": "A dog has 6 legs.",
                "nonsense": "Dogs don't have legs."
            }
        },
    ]

    # Build conversations for batch processing
    batch_convs = []
    test_info = []  # Track which test case each conversation belongs to

    for test_case in test_cases:
        prompt = test_case["prompt"]
        reference = test_case["reference"]

        # Include reference in the prompt to help model evaluate
        prompt_with_ref = (
                f"{prompt}\n\n"
                f"Reference answer: {reference}\n\n"
                f"Compare the assistant's response with the reference answer above. "
                f"Score based on how closely the response matches the reference in meaning and content."
            )

        for response_type, response in test_case["responses"].items():
            conv = [
                {"role": "user", "content": prompt_with_ref},
                {"role": "assistant", "content": response}
            ]
            batch_convs.append(conv)
            test_info.append({
                "prompt": prompt,
                "reference": reference,
                "response_type": response_type,
                "response": response
            })

    print(f"Processing {len(batch_convs)} conversations ({len(test_cases)} test cases)...")

    # Measure inference time
    start_time = time.time()
    batch_rewards = client(batch_convs)
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Inference time: {inference_time:.3f} seconds")
    print(f"Average time per conversation: {inference_time / len(batch_convs):.3f} seconds")
    print(f"Throughput: {len(batch_convs) / inference_time:.2f} conversations/second")

    if batch_rewards:
        # Apply sigmoid normalization (temperature=2.0)
        temperature = 2.0
        print("temperature: ", temperature)
        normalized_rewards = [1.0 / (1.0 + np.exp(-float(r) / temperature)) for r in batch_rewards]

        print("\nResults (Raw | Normalized):")
        print("-" * 80)
        for i, (info, raw, norm) in enumerate(zip(test_info, batch_rewards, normalized_rewards)):
            print(f"[{info['response_type'].upper():8s}] Raw: {raw:7.3f} | Norm: {norm:.4f}")
            print(f"  Q: {info['prompt']}")
            print(f"  A: {info['response']}")
            print()

        # Statistics by quality level
        quality_levels = ["perfect", "good", "partial", "wrong", "nonsense"]
        print("\nStatistics by Quality Level:")
        print("-" * 80)
        for quality in quality_levels:
            raw_scores = [batch_rewards[i] for i, info in enumerate(test_info) if info["response_type"] == quality]
            norm_scores = [normalized_rewards[i] for i, info in enumerate(test_info) if info["response_type"] == quality]
            if raw_scores:
                print(f"{quality.upper():8s}: Raw [{min(raw_scores):7.3f}, {max(raw_scores):7.3f}] avg={np.mean(raw_scores):7.3f} | "
                      f"Norm [{min(norm_scores):.4f}, {max(norm_scores):.4f}] avg={np.mean(norm_scores):.4f}")

        print(f"\nOverall Raw Range: [{min(batch_rewards):.3f}, {max(batch_rewards):.3f}]")
        print(f"Overall Normalized Range: [{min(normalized_rewards):.4f}, {max(normalized_rewards):.4f}]")
    else:
        print("Failed to get rewards.")
