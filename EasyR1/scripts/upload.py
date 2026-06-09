#!/usr/bin/env python3
"""
Upload model to Hugging Face Hub
"""
import argparse
import importlib.util
import os
from pathlib import Path

from huggingface_hub import create_repo, upload_folder


def upload_model_to_hf(
    model_path: str,
    repo_name: str,
    hf_token: str,
    repo_type: str = "model",
    private: bool = False,
    commit_message: str = "Upload model",
    inference_only: bool = False
):
    """
    Upload a model to Hugging Face Hub

    Args:
        model_path: Local path to the model directory
        repo_name: Repository name on HF Hub (format: username/repo-name)
        hf_token: Hugging Face API token
        repo_type: Type of repository (default: "model")
        private: Whether to create a private repository
        commit_message: Commit message for the upload
        inference_only: If True, only upload inference files (exclude training logs/results)
    """
    # Validate model path
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise ValueError(f"Model path does not exist: {model_dir}")

    if not model_dir.is_dir():
        raise ValueError(f"Model path is not a directory: {model_dir}")

    print(f"Model path: {model_dir}")
    print(f"Repository: {repo_name}")
    print(f"Private: {private}")

    # Create repository if it doesn't exist
    try:
        print(f"\nCreating repository: {repo_name}")
        create_repo(
            repo_id=repo_name,
            token=hf_token,
            repo_type=repo_type,
            private=private,
            exist_ok=True
        )
        print(f"Repository created/verified: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        raise

    # Upload the model folder
    try:
        print(f"\nUploading model from {model_dir}...")

        # Base ignore patterns
        ignore_patterns = ["*.git*", "__pycache__", "*.pyc", ".DS_Store", "README.md"]

        # Add training-related files to ignore if inference_only is True
        if inference_only:
            training_patterns = [
                "trainer_*.json*",
                "training_*.bin",
                "training_*.png",
                "train_results.json",
                "all_results.json",
                "checkpoint-*",
                "runs/",
                "*.log"
            ]
            ignore_patterns.extend(training_patterns)
            print("Inference-only mode: excluding training logs and checkpoints")

        upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_name,
            repo_type=repo_type,
            token=hf_token,
            commit_message=commit_message,
            ignore_patterns=ignore_patterns
        )

        print(f"\n✓ Model successfully uploaded to: https://huggingface.co/{repo_name}")

    except Exception as e:
        print(f"\n✗ Error uploading model: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Upload a model
            python upload.py --model_path /path/to/model --repo_name username/model-name --token YOUR_HF_TOKEN

            # Upload as a private repository
            python upload.py --model_path /path/to/model --repo_name username/model-name --token YOUR_HF_TOKEN --private

            # Use token from environment variable
            export HF_TOKEN=your_token_here
            python upload.py --model_path /path/to/model --repo_name username/model-name
        """
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory to upload"
    )

    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="Repository name on HF Hub (format: username/repo-name)"
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN environment variable)"
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )

    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload model",
        help="Commit message for the upload"
    )

    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Only upload inference files (exclude training logs, checkpoints, etc.)"
    )

    args = parser.parse_args()

    # Get token from args or environment
    hf_token = args.token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF token is required. Provide it via --token argument or HF_TOKEN environment variable"
        )

    # Upload the model
    upload_model_to_hf(
        model_path=args.model_path,
        repo_name=args.repo_name,
        hf_token=hf_token,
        private=args.private,
        commit_message=args.commit_message,
        inference_only=args.inference_only
    )


if __name__ == "__main__":
    main()
