# agent/ — the agent core

This folder holds the agent's core: the model-client seam and the tool-use loop.

- This app is a **tool-use agent**, not a one-shot script. The loop: send the
  user request + the available tool specs to the model (Bedrock Converse); if
  the model returns a `toolUse` block, run the matching tool and feed the
  `toolResult` back; repeat until the model returns a final text answer. Cap the
  number of turns so a confused model can't loop forever.
- Put the boto3 client behind `get_client()` / `get_model_id()` in
  `agent/client.py` — the only place that imports boto3 (the provider swap point).
- The loop discovers every tool in `tools/` automatically; do not hard-code or
  name tools one by one.
