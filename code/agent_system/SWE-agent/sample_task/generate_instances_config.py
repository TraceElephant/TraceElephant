#!/usr/bin/env python3
"""
Generate SWE-agent task configuration file from parquet file

Usage:
    python generate_instances_config.py <parquet_file> --total 100 --output instances.json

The generated file can be used like this:
    sweagent run-batch \\
        --config config/default.yaml \\
        --agent.model.name gpt-4o \\
        --instances.type file \\
        --instances.path instances.json
"""

import pandas as pd
import argparse
from collections import Counter
import json
from pathlib import Path


def generate_instances_config(
    parquet_file: str,
    total_samples: int = 100,
    output_file: str = "instances.json",
    report_file: str = None
):
    """
    Generate SWE-agent instance configuration file from parquet file

    Args:
        parquet_file: Path to parquet data file
        total_samples: Total number of samples (default 100)
        output_file: Output configuration file path (JSON format)
        report_file: Output report file path (optional, for viewing selected tasks)
    """
    try:
        # -------------------------- 1. Read data and validate --------------------------
        print("Reading parquet file...")
        df = pd.read_parquet(parquet_file)

        # Validate required fields
        required_cols = ['repo', 'instance_id', 'problem_statement', 'base_commit']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Parquet file missing required fields: {', '.join(missing_cols)}")

        # -------------------------- 2. Count original instances per repo --------------------------
        print("Counting original instances per repo...")
        repo_counts = Counter(df['repo'])
        sorted_repos = sorted(repo_counts.items(), key=lambda x: x[1], reverse=True)
        total_original = sum(repo_counts.values())

        # -------------------------- 3. Calculate sampling quota (ensure at least 1 per repo) --------------------------
        print("Calculating sampling quota...")
        repo_quota = {}
        total_repos = len(sorted_repos)
        remaining_quota = total_samples - total_repos  # Allocate 1 per repo first

        # Calculate ratio for each repo
        repo_ratios = {repo: count / total_original for repo, count in repo_counts.items()}

        # Allocate remaining quota (proportionally rounded)
        temp_quota = {}
        for repo in repo_counts.keys():
            temp_quota[repo] = int(round(repo_ratios[repo] * remaining_quota))

        # Calibrate quota error
        quota_diff = remaining_quota - sum(temp_quota.values())
        if quota_diff != 0:
            top_repo = max(repo_counts.items(), key=lambda x: x[1])[0]
            temp_quota[top_repo] += quota_diff

        # Final quota = base 1 + proportional quota
        for repo in repo_counts.keys():
            repo_quota[repo] = 1 + temp_quota[repo]

        # Final calibration of total quota (edge cases)
        final_total = sum(repo_quota.values())
        if final_total != total_samples:
            diff = total_samples - final_total
            top_repo = max(repo_counts.items(), key=lambda x: x[1])[0]
            repo_quota[top_repo] += diff

        # -------------------------- 4. Select instances and generate config --------------------------
        print("Selecting instances and generating config...")
        instances = []
        sampled_data = []  # For report

        for repo, quota in repo_quota.items():
            # Filter instances for this repo, sort by instance_id (ensure reproducibility)
            repo_df = df[df['repo'] == repo].sort_values('instance_id')
            # Take first quota entries
            selected_df = repo_df.head(quota)

            for _, row in selected_df.iterrows():
                # Generate docker image name (following SWE-bench convention)
                iid = row['instance_id']
                id_docker_compatible = iid.replace("__", "_1776_")
                image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()

                # Create instance config (SimpleBatchInstance format)
                instance = {
                    "instance_id": iid,
                    "problem_statement": row['problem_statement'].strip(),
                    "image_name": image_name,
                    "repo_name": "testbed",  # SWE-bench standard repo name
                    "base_commit": row['base_commit']
                }

                # If image_assets field exists (multimodal), add extra fields
                if 'image_assets' in row and pd.notna(row['image_assets']):
                    try:
                        import json as _json
                        image_assets = _json.loads(row['image_assets'])
                        if 'problem_statement' in image_assets:
                            instance['issue_images'] = image_assets['problem_statement']
                    except:
                        pass

                instances.append(instance)

                # Collect report info
                sampled_data.append({
                    'repo': repo,
                    'instance_id': iid,
                    'problem_statement': row['problem_statement'].strip()
                })

        # -------------------------- 5. Save config file --------------------------
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(instances, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Config file generated: {output_file}")
        print(f"   - Contains {len(instances)} task instances")
        print(f"   - Involves {total_repos} repositories")

        # -------------------------- 6. Generate readable report (optional) --------------------------
        if report_file:
            generate_report(
                report_file,
                sampled_data,
                repo_counts,
                repo_quota,
                total_original,
                total_repos,
                output_file
            )
            print(f"✅ Report file generated: {report_file}")

        # -------------------------- 7. Output usage example --------------------------
        print(f"\n📝 Usage:")
        print(f"""
sweagent run-batch \\
    --config config/default.yaml \\
    --agent.model.name gpt-4o \\
    --num_workers 3 \\
    --agent.model.per_instance_cost_limit 3.00 \\
    --instances.type file \\
    --instances.path {output_file}
        """.strip())

    except FileNotFoundError:
        print(f"❌ Error: File '{parquet_file}' not found")
    except Exception as e:
        print(f"❌ Error occurred during processing: {str(e)}")
        import traceback
        traceback.print_exc()


def generate_report(
    report_file: str,
    sampled_data: list,
    repo_counts: Counter,
    repo_quota: dict,
    total_original: int,
    total_repos: int,
    config_file: str
):
    """Generate readable report file"""
    report_content = []
    report_content.append("="*100)
    report_content.append("SWE-bench Task Sampling Report")
    report_content.append("="*100)

    # Basic statistics
    report_content.append("\n[Basic Statistics]")
    report_content.append(f"- Config file: {config_file}")
    report_content.append(f"- Original dataset total instances: {total_original}")
    report_content.append(f"- Total repositories: {total_repos}")
    report_content.append(f"- Total sampled instances: {len(sampled_data)}")

    # Original instances per repo
    report_content.append("\n[Original Instances Per Repository]")
    report_content.append("-"*80)
    report_content.append(f"{'Repository':<40} {'Original Count':<15} {'Ratio':<10}")
    report_content.append("-"*80)
    sorted_repos = sorted(repo_counts.items(), key=lambda x: x[1], reverse=True)
    for repo, count in sorted_repos:
        ratio = count / total_original * 100
        report_content.append(f"{repo:<40} {count:<15} {ratio:>6.2f}%")

    # Sampling quota per repo
    report_content.append("\n[Sampling Quota Per Repository]")
    report_content.append("-"*100)
    report_content.append(f"{'Repository':<40} {'Sampled':<10} {'% of Original':<20} {'% of Total Sample':<20}")
    report_content.append("-"*100)
    sorted_quota = sorted(repo_quota.items(), key=lambda x: x[1], reverse=True)
    for repo, quota in sorted_quota:
        original_count = repo_counts[repo]
        ratio_original = (quota / original_count) * 100
        ratio_total = (quota / len(sampled_data)) * 100
        report_content.append(
            f"{repo:<40} {quota:<10} {ratio_original:>6.2f}%{' '*12} {ratio_total:>6.2f}%"
        )

    # Detailed info of selected instances
    report_content.append("\n[Selected Task Instances]")
    report_content.append("="*100)

    # Display grouped by repo
    from collections import defaultdict
    instances_by_repo = defaultdict(list)
    for data in sampled_data:
        instances_by_repo[data['repo']].append(data)

    for repo in sorted(instances_by_repo.keys()):
        instances = instances_by_repo[repo]
        report_content.append(f"\n## Repository: {repo} ({len(instances)} tasks)")
        report_content.append("-"*100)

        for idx, data in enumerate(instances, 1):
            report_content.append(f"\n  [{idx}] {data['instance_id']}")
            # Format task description (line breaks for readability)
            desc_lines = data['problem_statement'].split('\n')
            report_content.append(f"      Description: {desc_lines[0][:80]}...")
            if len(desc_lines) > 1:
                report_content.append(f"                   {desc_lines[1][:80]}...")

    # Write report file
    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate SWE-agent task configuration file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Generate config file with 100 tasks
  python generate_instances_config.py data/swe_bench_verified.parquet --total 100

  # Generate config file and report
  python generate_instances_config.py data/swe_bench_verified.parquet \\
      --total 50 \\
      --output my_instances.json \\
      --report my_report.txt

  # Run SWE-agent with generated config file
  sweagent run-batch \\
      --config config/default.yaml \\
      --agent.model.name gpt-4o \\
      --instances.type file \\
      --instances.path my_instances.json
        """
    )
    parser.add_argument('parquet_file', help='Path to the parquet file')
    parser.add_argument('--total', type=int, default=100,
                        help='Total number of samples (default: 100)')
    parser.add_argument('--output', type=str, default="instances.json",
                        help='Output config file path (default: instances.json)')
    parser.add_argument('--report', type=str, default=None,
                        help='Output report file path (optional, for viewing selected tasks)')

    args = parser.parse_args()

    # If no report file specified, auto-generate one
    if args.report is None:
        output_stem = Path(args.output).stem
        args.report = f"{output_stem}_report.txt"

    generate_instances_config(
        args.parquet_file,
        args.total,
        args.output,
        args.report
    )
