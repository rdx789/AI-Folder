#!/usr/bin/env python3
"""Customer support agent CLI — smoke-tests LLM credentials and the tool-use loop.

Usage:
    # Run a named scenario from the registry:
    python main.py --scenario angry_refund_request

    # Pass a custom message directly:
    python main.py "I can't log in and my reset email never arrived."

    # List available scenarios:
    python main.py --list
"""
import argparse
import sys

from agent.loop import run
from prompts.registry import SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Customer support agent — smoke-tests LLM credentials and the tool-use loop."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--scenario", "-s",
        metavar="NAME",
        help="Run a built-in scenario by name (see --list for options).",
    )
    group.add_argument(
        "--list", "-l",
        action="store_true",
        help="Print available scenario names and exit.",
    )
    parser.add_argument(
        "message",
        nargs="?",
        help="Custom support message to send to the agent.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for name, mod in SCENARIOS.items():
            print(f"  {name:<25} {mod.DESCRIPTION}")
        sys.exit(0)

    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario {args.scenario!r}. Run --list to see options.", file=sys.stderr)
            sys.exit(1)
        user_message = SCENARIOS[args.scenario].PROMPT
    elif args.message:
        user_message = args.message
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Customer: {user_message}\n")
    print("Agent: ", end="", flush=True)
    try:
        response = run(user_message)
    except Exception as e:
        print(f"\nAPI key test FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    print(response)
    print("\nAPI key test PASSED — the model responded successfully.")


if __name__ == "__main__":
    main()
