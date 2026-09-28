<!-- Rename this file to CLAUDE.md so your coding agent picks it up. -->

# Conventions for this build

- Every model call goes through Bedrock via `langchain_aws.ChatBedrockConverse`. Isolate
  it in one place; don't create clients scattered through the code.
- Load config with `load_dotenv(find_dotenv())`, and read `BEDROCK_MODEL_ID` /
  `AWS_REGION` from the environment. Never hardcode a model id, a region, or a key.
- Tools come from the running MCP server over `streamable_http` — discover them at
  startup, never reimplement them.
- Anything already in this repo is provided infrastructure: read it and import from it
  rather than rewriting it.
- Comment the non-obvious: why a parameter, why a threshold, why this and not the
  alternative. Skip comments that restate the code.
