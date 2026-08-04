Build the personal-finance-analyst agent described by the CLAUDE.md files in this
repo. Follow every contract already written down — don't reinterpret or relax them.

1. **Data.** Invoke the `generate-synthetic-data` skill to generate the dataset in
   `data/` per the contract in `data/CLAUDE.md`. Run its validation pass and confirm
   it's green before moving on.
2. **Agent core.** Build `agent/` per `agent/CLAUDE.md`: the Bedrock client seam
   (`get_client()` / `get_model_id()`), and the tool-use loop (send request + tool
   specs, run any `toolUse` the model returns, feed back `toolResult`, repeat with a
   turn cap) that auto-discovers every tool in `tools/`.
3. **Prompts.** Build `prompts/` per `prompts/CLAUDE.md`: the system prompt, and the
   sample user prompts as a registry the agent can list and select from.
4. **Tools.** Build `tools/` per `tools/CLAUDE.md`, creating every tool with the
   `create-agent-tool` skill (one invocation per tool) — it schemas via
   `schema-smith`, writes the handler + tests, and registers each tool. Tools read
   from `data/`, not mocks.
5. **Wire it up.** Add a `main.py` (or equivalent CLI entrypoint) to list and run
   scenarios end to end, plus a `requirements.txt` for whatever you import. Load
   config from the existing `.env`.
6. **Prove it works.** Run the agent against at least one sample prompt that can
   only be answered by querying `data/`, and show the tool-call trace plus the
   final answer.

Do not ask me to design the dataset schema, the tools, or the agent's behavior —
that's already specified in the CLAUDE.md files. Report what you built, the test
results, and the live run's tool trace when done.
