#!/usr/bin/env python3
"""
Main script to run Training-Free GRPO for Earth-Agent

Usage:
    python run_training_free_grpo.py --config training_free_grpo/configs/example_config.json

    # Or with command-line overrides:
    python run_training_free_grpo.py \
        --config training_free_grpo/configs/example_config.json \
        --epochs 2 \
        --batch_size 20 \
        --grpo_n 5
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training_free_grpo.config import TrainingFreeGRPOConfig
from training_free_grpo.training_free_grpo import TrainingFreeGRPO


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run Training-Free GRPO for Earth-Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration JSON file'
    )

    # Override arguments
    parser.add_argument('--exp_id', type=str, help='Experiment ID override')
    parser.add_argument('--epochs', type=int, help='Number of epochs override')
    parser.add_argument('--batch_size', type=int, help='Batch size override')
    parser.add_argument('--grpo_n', type=int, help='Number of rollouts per question override')
    parser.add_argument('--rollout_temperature', type=float, help='Rollout temperature override')
    parser.add_argument('--restart_step', type=int, help='Restart from this step (None=use cache, 0=restart all)')
    parser.add_argument('--question_ids', type=str, nargs='+', help='List of question IDs to use')
    parser.add_argument('--truncate', type=int, help='Limit number of questions (for debugging)')

    return parser.parse_args()


def load_and_override_config(args) -> TrainingFreeGRPOConfig:
    """Load config from JSON and apply command-line overrides"""
    # Load base configuration
    config = TrainingFreeGRPOConfig.from_json(args.config)

    # Apply overrides
    if args.exp_id:
        config.exp_id = args.exp_id

    if args.epochs:
        config.practice.epochs = args.epochs

    if args.batch_size:
        config.practice.batch_size = args.batch_size

    if args.grpo_n:
        config.practice.grpo_n = args.grpo_n

    if args.rollout_temperature:
        config.practice.rollout_temperature = args.rollout_temperature

    if args.restart_step is not None:
        config.practice.restart_step = args.restart_step if args.restart_step > 0 else 0

    if args.question_ids:
        config.question_ids = args.question_ids

    if args.truncate:
        config.practice.rollout_data_truncate = args.truncate

    return config


async def main():
    """Main execution function"""
    # Parse arguments
    args = parse_args()

    # Load configuration
    print("Loading configuration...")
    config = load_and_override_config(args)

    print("\n" + "="*80)
    print("Training-Free GRPO for Earth-Agent")
    print("="*80)
    print(f"Configuration:")
    print(f"  Experiment ID: {config.exp_id}")
    print(f"  Epochs: {config.practice.epochs}")
    print(f"  Batch size: {config.practice.batch_size}")
    print(f"  GRPO-N: {config.practice.grpo_n}")
    print(f"  Rollout temperature: {config.practice.rollout_temperature}")
    print(f"  Concurrency: {config.practice.rollout_concurrency}")
    print(f"  Output directory: {config.output_dir}")
    if config.question_ids:
        print(f"  Question IDs: {len(config.question_ids)} questions")
    if config.practice.rollout_data_truncate:
        print(f"  Truncated to: {config.practice.rollout_data_truncate} questions")
    print("="*80 + "\n")

    # Create and run Training-Free GRPO
    grpo = TrainingFreeGRPO(config)
    enhanced_config_path = await grpo.run()

    print("\n" + "="*80)
    print("Training completed successfully!")
    print(f"Enhanced agent configuration: {enhanced_config_path}")
    print("\nTo evaluate the enhanced agent, use:")
    print(f"  python evaluate_enhanced_agent.py --config {enhanced_config_path}")
    print("="*80)


if __name__ == "__main__":
    # Set environment variables if needed
    os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")

    # Run async main
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
