#!/usr/bin/env python3
"""HuggingFace model downloader for SWAY"""

from huggingface_hub import snapshot_download, hf_hub_download, list_models
import argparse
import os


def search_family(family, limit=5):
    """Search for models from a specific family"""
    print(f"\nSearching for {family} models...")
    print("=" * 60)

    try:
        models = list_models(search=family, limit=limit)
        for i, model in enumerate(models, 1):
            print(f"{i}. {model.id}")
            print(f"   Downloads: {model.downloads:,}")
        return models
    except Exception as e:
        print(f"Error searching for {family}: {e}")
        return []


def download_model(repo_id, local_dir=None, token=None):
    """Download a model from HuggingFace Hub"""
    print(f"Downloading {repo_id}...")

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token,
        resume_download=True,
    )

    print(f"✓ Downloaded to: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from HuggingFace Hub")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for models by family")
    search_parser.add_argument("family", help="Model family (e.g., qwen, gemma, mistral)")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("repo_id", help="HuggingFace repo ID (e.g., mistralai/Mistral-7B)")
    download_parser.add_argument("--output", help="Local directory to save model")
    download_parser.add_argument("--token", help="HuggingFace API token (optional)")

    args = parser.parse_args()

    if args.command == "search":
        search_family(args.family, limit=args.limit)
    elif args.command == "download":
        download_model(args.repo_id, local_dir=args.output, token=args.token)
    else:
        parser.print_help()
