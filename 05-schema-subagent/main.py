"""CLI entry point — pick a sample support scenario and run it through the agent.

    python main.py --list
    python main.py --scenario order_status_check
"""
import argparse
import sys

from agent.loop import run_agent
from prompts.registry import get_scenario, list_scenarios
from prompts.system_prompt import SYSTEM_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a customer-support agent scenario.")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit.")
    parser.add_argument("--scenario", help="Name of the scenario to run.")
    args = parser.parse_args()

    scenarios = list_scenarios()

    if args.list or not args.scenario:
        print("Available scenarios:")
        for name in scenarios:
            print(f"  - {name}")
        if not args.scenario:
            return

    if args.scenario not in scenarios:
        parser.error(f"unknown scenario '{args.scenario}'. Choices: {', '.join(scenarios)}")

    user_prompt = get_scenario(args.scenario)
    print(f"--- scenario: {args.scenario} ---\n{user_prompt}\n")
    try:
        answer = run_agent(SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        print(f"agent run failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"--- agent response ---\n{answer}")


if __name__ == "__main__":
    main()
