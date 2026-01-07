import os
import argparse
import sys
import datetime
import json
from dotenv import load_dotenv
from openai import OpenAI

from lib.utils_parallel import (
    all_at_once_parallel,
    step_by_step_parallel,
    binary_search_parallel
)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Analyze multi-agent chat history using specific models.")

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["all_at_once", "step_by_step", "binary_search"],
        help="The analysis method to use."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (OpenAI-compatible)."
    )
    parser.add_argument(
        "--directory_path",
        type=str,
            default = "../../data_dir",
            help="Path to the directory containing JSON chat history files. Default: '../../data_dir'."
    )

    parser.add_argument(
        "--api_key", type=str, default="sk-xxx",
        help="OpenAI-compatible API key."
    )
    parser.add_argument(
        "--base_url", type=str, default="https://api.xxx.com/v1",
        help="OpenAI-compatible base URL."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096,
        help="Maximum number of tokens for GPT API response. Used only for GPT models."
    )

    parser.add_argument(
        "--max_workers", type=int, default=10,
        help="Maximum number of parallel workers."
    )

    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Output directory for results. Default: 'outputs'."
    )

    args = parser.parse_args()

    client_or_model_obj = None

    if not args.api_key:
        print("Error: --api_key is required for OpenAI-compatible API calls")
        sys.exit(1)

    try:
        client_or_model_obj = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url
        )
        print(f"Successfully initialized OpenAI client for base_url: {args.base_url}")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        sys.exit(1)


    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{args.method}_{args.model.replace('/','_')}.jsonl"
    output_filepath = os.path.join(output_dir, output_filename)

    # Adjust default worker count based on method
    if args.max_workers == 10 and args.method in ["step_by_step", "binary_search"]:
        args.max_workers = 5

    print(f"Analysis method: {args.method}")
    print(f"Model: {args.model}")
    print(f"Parallel mode: Enabled (max_workers={args.max_workers})")
    print(f"Output will be saved to: {output_filepath}")

    try:
        print(f"--- Starting Analysis: {args.method} ---")
        print(f"Timestamp: {datetime.datetime.now()}")
        print(f"Model Used: {args.model}")
        print(f"Input Directory: {args.directory_path}")
        print("-" * 20)

        results = []

        # Use parallel processing
        print(f"Using parallel processing with {args.max_workers} workers")
        if args.method == "all_at_once":
            results = all_at_once_parallel(
                client=client_or_model_obj,
                directory_path=args.directory_path,
                model=args.model,
                max_tokens=args.max_tokens,
                max_workers=args.max_workers
            )
        elif args.method == "step_by_step":
            results = step_by_step_parallel(
                client=client_or_model_obj,
                directory_path=args.directory_path,
                model=args.model,
                max_tokens=args.max_tokens,
                max_workers=args.max_workers
            )
        elif args.method == "binary_search":
            results = binary_search_parallel(
                client=client_or_model_obj,
                directory_path=args.directory_path,
                model=args.model,
                max_tokens=args.max_tokens,
                max_workers=args.max_workers
            )

        for record in results:
            record["method"] = args.method
            record["model"] = args.model

        with open(output_filepath, 'w', encoding='utf-8') as output_file:
            for record in results:
                output_file.write(json.dumps(record, ensure_ascii=True) + "\n")

        print("-" * 20)
        print(f"--- Analysis Complete ---")
        print(f"Analysis finished. Output saved to {output_filepath}")

    except Exception as e:
        print(f"\n!!! An error occurred during analysis or file writing: {e} !!!", file=sys.stderr)
  
if __name__ == "__main__":
    main()
