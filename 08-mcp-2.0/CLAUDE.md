<!-- Rename this file to CLAUDE.md to activate these conventions, then build from PROMPT.md (in the reference repo's SDD/ folder). -->

# CLAUDE.md — build conventions

- The provided eval scorers — `eval/checks.py` and `eval/judges.py` — are a fixed dependency: import and call them; never edit or reimplement them.
- Load all configuration from the environment via `load_dotenv(find_dotenv())`; never hard-code a value. Before running, confirm every required variable is set, and if one is missing, STOP and ask rather than guess or default. Required: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `BEDROCK_MODEL_ID`, `BEDROCK_EMBEDDING_MODEL_ID`. (No `.env.example` ships here — bring the `.env` you used elsewhere in this course.)
- Make every LLM call through boto3 Bedrock Converse, and keep each call in one small function so the provider stays swappable.
- When you need structured output from a model, force a tool call instead of parsing prose.
- Comment only non-obvious AI/SDK decisions; leave self-evident code uncommented.
