# CLAUDE.md — conventions for this project

- Call the model with boto3 Bedrock Converse.
- Keep the Bedrock call in one function so a provider swap is easy.
- Config from env: `.env` via `load_dotenv(find_dotenv())`; model ID + region from env vars, never hard-coded.
- Comment only the non-obvious AI/SDK bits — skip the obvious.
