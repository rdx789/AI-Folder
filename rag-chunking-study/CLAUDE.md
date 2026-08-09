<!-- Rename this file to CLAUDE.md to activate it, then build rag_tool.py from the lesson's PROMPT.md (in the lesson repo's SDD/ folder). -->

# CLAUDE.md — conventions for this rebuild

- Call the model with boto3 Bedrock Converse.
- Keep the Bedrock call in one function so a provider swap is easy.
- Config from env: `.env` via `load_dotenv(find_dotenv())`; model IDs + region from env vars, never hard-coded.
- Anchor file paths to the script location, not the shell's cwd, so a script runs from anywhere.
- Comment only the non-obvious AI/SDK bits — skip the obvious.
