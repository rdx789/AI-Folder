# prompts/ — prompts as Python, one per file

Every prompt is its **own Python file** — not markdown. Python gives you real
templating: parameters, f-strings, helper functions, and a clean import surface
as prompts grow. Each file exposes its prompt as a module-level string, or a
function that builds and returns one.

- **One prompt per file.** Never pack multiple prompts into a single file.
- **Write at least 5 sample user prompts**, one per file — each a realistic
  customer-support scenario to test the LLM against. Don't ship fewer. For
  example: an angry refund request, a "where is my order?" status check, a
  password-reset how-to, a billing dispute, a product-feature question.
- Make the sample prompts **discoverable and selectable**: expose them so the
  agent can list the available scenarios and **choose** one to run — e.g. a
  registry mapping a short name to each prompt. Dropping in a new file and
  registering it is all it takes to add a scenario.
- Also include the **system prompt** here, in its own file, framing the model as
  a customer support assistant that uses its tools rather than guessing.
- No business logic in these files beyond building the prompt string — the loop
  and tool-calling live in `agent/`.
