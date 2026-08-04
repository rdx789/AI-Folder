"""CLI entrypoint: list or run personal-finance-analyst scenarios."""
import argparse
import json
import sys

from agent.loop import run_agent
from prompts.registry import get_scenario, list_scenarios
from prompts.system_prompt import SYSTEM_PROMPT


def print_tool_call(entry):
    if "result" in entry:
        print(f"  -> tool call: {entry['tool']}({entry['input']})")
        print(f"     result: {json.dumps(entry['result'])[:500]}")
    else:
        print(f"  -> tool call: {entry['tool']}({entry['input']}) ERROR: {entry['error']}")


def run_scenario(name):
    prompt = get_scenario(name)
    print(f"Scenario: {name}")
    print(f"User prompt: {prompt}\n")
    final_text, trace = run_agent(SYSTEM_PROMPT, prompt, on_tool_call=print_tool_call)
    print("\nFinal answer:")
    print(final_text)


def main():
    parser = argparse.ArgumentParser(description="Personal finance analyst agent")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available scenarios")

    run_parser = sub.add_parser("run", help="Run a scenario by name")
    run_parser.add_argument("scenario", help="Scenario name (see `main.py list`)")

    args = parser.parse_args()

    if args.command == "list":
        for name in list_scenarios():
            print(name)
    elif args.command == "run":
        if args.scenario not in list_scenarios():
            print(f"Unknown scenario: {args.scenario!r}. Run `main.py list` for options.", file=sys.stderr)
            sys.exit(1)
        run_scenario(args.scenario)


if __name__ == "__main__":
    main()
