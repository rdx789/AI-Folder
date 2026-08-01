"""CLI: test that the LLM API key works for a customer support scenario.

Usage:
    python main.py --list
    python main.py <scenario-name>
"""

import argparse
import sys

from agent.loop import run
from prompts.scenarios import get_prompt, list_scenarios
from prompts.system_prompt import SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario", nargs="?", help="Scenario name to run (see --list)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available scenarios and exit"
    )
    args = parser.parse_args()

    scenarios = list_scenarios()

    if args.list or not args.scenario:
        print("Available scenarios:")
        for name, description in scenarios.items():
            print(f"  {name:<26} {description}")
        if not args.scenario:
            sys.exit(0 if args.list else 1)

    if args.scenario not in scenarios:
        print(f"Unknown scenario: {args.scenario!r}. Use --list to see options.")
        sys.exit(1)

    prompt = get_prompt(args.scenario)
    print(f"--- Scenario: {args.scenario} ---")
    print(f"User: {prompt}\n")

    try:
        answer = run(prompt, SYSTEM_PROMPT)
    except Exception as e:
        print(f"LLM call failed: {e}")
        sys.exit(1)

    print(f"Model: {answer}")


if __name__ == "__main__":
    main()
