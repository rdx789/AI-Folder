<!-- Reverse-engineered on 2026-10-01 from the final, verified Maya code so that handing
     this prompt (plus the provided inputs listed below) to a coding agent on a new machine
     rebuilds an equivalent, working project. Plain text is the prompt; comments like this
     one are notes for you and are not part of it.

     Origin: Lesson 10 homework ("Build the Maya onboarding evidence agent"), the first
     flagship workflow of the NovaOps final project. The numbers in "Acceptance" are what
     the original build measured; treat them as targets, not as values to hard-code. -->

# Build: Maya, a caller-safe onboarding evidence agent for NovaOps

Sara Ben-David (`E004`, group `UG_HR`) is onboarding Maya Cohen (`E001`), a Customer
Success Manager starting 2026-08-01, hybrid, in Israel. Build an agent that answers her
with an **evidence-backed, structured onboarding checklist** — every claim an exact quote
from a cited source — across a twelve-turn conversation (`S2-onboarding-maya`) and in a
single turn when she asks for the whole checklist at once:

> Prepare an evidence-backed onboarding checklist for Maya Cohen, a Customer Success
> Manager starting 2026-08-01. Include required systems, equipment, policy
> acknowledgements, and anything currently blocked.

The agent reads documents and records; it never writes. The one action it can cause —
requesting Webex access, which is blocked because 42 seats are used against a 40-seat
limit — goes through a typed port to a separate workflow, exactly once, and is reported
as **pending**, never granted.

## Provided inputs (keep; do not regenerate)

- `data/` — the NovaOps dataset: Markdown in `employment/`, `policies/`, `it_kb/`,
  `memos/` (optionally `contracts/`), and `database/schema.sql` + `seed.sql` (employees,
  onboarding_tasks, assets, tickets, systems, vendors, software_subscriptions).
- `code/access_manifest.json` — reviewed access metadata for every Markdown source,
  keyed by root-relative path: `audience` (subset of `UG_HR`, `UG_IT`, `UG_REGULAR`),
  `sensitivity` (`internal|confidential|restricted`), `subject_employee_id` (or absent),
  `self_service` (bool), `allowed_employee_ids` (list), optional `updated_at`.
- `code/evals/data/sessions.json` — the S2 session: `caller` (`E004/UG_HR`) and 12 ordered
  turns, each with `user`, `expected_intent`, `expected_tools`, `forbidden_tools`,
  `required_facts` (`a|b` = alternatives), `should_use_tools`, optional `arg_checks`;
  turn 10 forbids `create_access_request` and expects one `webex_access` handoff
  (`employee_id E001`, `system Webex`, `status pending`).
- `code/evals/data/s2_maya.json` — per-turn `expected_read_tools`, `requires_retrieval`,
  `expected_handoff` for the tool-reachability matrix.

## Stack and layout

Python ≥ 3.10, LangGraph (checkpointed `StateGraph`), Pydantic `TypeAdapter`, boto3
(Bedrock Converse with a **forced** `structured_result` tool for every model output;
Titan Text Embeddings v2, 1024-d, normalized), opensearch-py (AWS SigV4; Serverless
`aoss` or managed `es`), MCP Python SDK (FastMCP server, stdio/HTTP client),
python-dotenv. setuptools packaging:

```text
maya/                project root: .env, .env.example, pyproject.toml, requirements.txt, setup.sh
  code/              importable as package `maya` (pyproject: package-dir = {maya = "code"})
  data/              default DATA_PATH
  results/           run outputs (git-ignored)
```

`code/paths.py` is the only place that knows locations: `CODE_DIR`, `PROJECT_DIR`
(parent), `ENV_FILE` (`PROJECT_DIR/.env`), `RESULTS_DIR`, and `DATA_PATH` resolved as env
`DATA_PATH` → `DATA_PATH=` read from `.env` (that single value only, so importing never
loads credentials) → legacy `NOVAOPS_DATA_ROOT` → `PROJECT_DIR/data`; relative values are
resolved against `PROJECT_DIR`. Nothing else may build a path to the data, `.env` or
results. Importing the package makes no network calls.

## Typed contracts (`schemas.py`, frozen dataclasses where noted)

- `CallerContext(employee_id, user_group)` — frozen; comes only from trusted runtime
  config `configurable.caller`, never from model output or user input.
- `EvidenceCitation(source, chunk_id, section, source_id, collection, audience,
  subject_employee_id, sensitivity, updated_at)`; `Evidence(text, citation)`.
- `ContextPlan(current_intent, subject_employee_id="E001", relevant_facts,
  active_constraints, required_evidence, retrieval_query, requires_retrieval,
  requires_operational_reads, also_needs)`.
- `ConversationMemory(current_intent, important_facts, active_constraints, decisions,
  unresolved_items, closed_topics)`.
- `ChecklistTask(description, status ∈ proposed|complete|blocked|pending|unresolved,
  evidence, task_id, support_quotes, reason)` — proposed/blocked require evidence.
- `OnboardingChecklist(employee_id, role, start_date, tasks, full_name, location,
  work_mode, active_constraints, evidence, status ∈ ok|unresolved|denied|pending, message,
  handoff, pending_access, related_status)` with a `blocked_items` property.
- `WebexHandoff(caller, employee_id, business_reason, idempotency_key, system="Webex",
  evidence, kind="webex_access")`; `PendingAccessResult(request_id, status="pending")`
  rejects anything but pending; `ReadToolCall(name, arguments)`;
  `ModelTurn(tool_calls, tasks, message)`.

## Access rules (`access.py`) — enforced before anything else

Groups `UG_HR`, `UG_IT`, `UG_REGULAR`; collections `employment, policies, it_kb,
contracts, memos`. Validate caller IDs as `E\d{3}`. `authorize_subject(caller, subject)`:
only HR may act on another employee; others only on themselves; anything else raises
`PermissionDenied`, whose single message is *"I can't share that information. Please
contact HR for assistance."* Denial happens **before** planning, embedding, search,
reranking, reads or handoff, and the denied response contains no employee facts, source
names, citations or counts. The hard OpenSearch filter = collection allowlist ∧
sensitivity allowlist ∧ (audience contains the caller's group ∨ (self_service ∧
subject = caller ∧ caller ∈ allowed_employee_ids)), ANDed with a soft subject filter
(subject = target employee, or no subject). Re-check every hit in code (`record_is_allowed`)
and fail closed on any unexpected or malformed hit.

## Ingestion (`ingestion.py`, `ingest.py`)

Only Markdown under the five collections; every source must be in the manifest or
ingestion fails closed; person-specific documents may not be visible to `UG_REGULAR` as a
group. Chunk 250 words with 50 overlap (the corpus yields 42 chunks). `source_id =
sha256(relative path)`; `chunk_id = sha256(source_id:n:chunk text)`. Dates come from the
manifest or explicit `Last updated|Date|Effective date:` lines, never file times. Index
mapping is `dynamic: strict` with keyword metadata, `updated_at` as date, and a
`knn_vector` using **faiss** HNSW inner product (Serverless NMSLIB cannot run inline
filters — refuse it). `ingest` skips unchanged evidence (compares text, metadata, vectors
and the embedding model stored in mapping `_meta`), `--force` re-embeds everything before
replacing the index, and nothing ever deletes the collection. Errors name the bad file.

## Retrieval (`retrieval.py`, `reranker.py`)

`EvidenceRetriever.retrieve(*, caller, plan, sources=())`. The OpenSearch adapter embeds
`query_for(plan)` (planner query, or intent + facts + constraints + required files; missing
required filenames appended), runs k-NN with the filter **inside** `knn.vector.filter`
(never post-filter), takes 20 candidates, reranks with a Bedrock tool call that must score
every candidate (`{"scores":[{"index","score"}]}`), and returns the top 4 as `Evidence`.
The reranker may only reorder: a returned list that adds, drops or alters candidates
raises; if reranking fails or is invalid, fall back to vector order of the already
ACL-checked candidates (count the failure). `retrieve_evidence_node` authorizes, runs one
round, and if `plan.required_evidence` is still missing allows **exactly one** refinement
that keeps the same ACL and subject clauses and adds an exact `terms` filter on `source`
(`<collection>/<file>` for each missing file) or `source_id`, returning the best passage
of **each** named document (limit = max(4, number of documents)).

## Operational reads (`operations.py`, `server/`)

Seven read-only tools, the entire catalogue: `get_employee(query)`,
`list_onboarding_tasks(employee_id)`, `check_asset_inventory(asset_type?, location?)`,
`list_policies()`, `get_policy(name)` (name ∈ policy stems), `check_software_subscription
(software)` (adds `seats_available`, `over_limit`), `list_employee_tickets(employee_id,
status?)`, over an in-memory SQLite built from `data/database`. Expose them through a
FastMCP server (`maya-server`, HTTP on 9878 or `MAYA_MCP_PORT`, `--port`) and a
`LocalReadPort`; an `MCPReadPort` wraps a connected client session. There is **no write
tool** — `create_access_request` must never exist in any loadout, fallback or catalogue.
Before execution, validate the whole batch: tool in the selected loadout, arguments match
the schema, and employee scope (E001; E010 only for an active Rachel-ticket question,
`list_employee_tickets` only for `ticket_status`). Sanitize results: `get_employee` rows
must all be the target (else deny); non-HR callers never see others' asset assignments.
Each read becomes an `Evidence` snapshot (`collection="operational"`,
`source="novaops/<tool>"`) that replaces the previous snapshot of the same lookup.

## Model seam (`model.py`, `answer_context.py`, `grounding.py`)

`PlanningModel` has `plan`, `distill`, `respond`; the Bedrock implementation uses
temperature 0 and a forced JSON-schema tool. Malformed responses raise `ValueError`
naming the `stopReason`.

- **plan** → `ContextPlan`; enums restrict `current_intent`/`also_needs` to the intents
  below and `required_evidence` to the real source filenames. Route by the data source the
  newest request needs, not the session theme; recall/summary turns need neither
  retrieval nor reads; filing an already supported Webex request needs neither.
- **distill** folds whole older turns into `ConversationMemory`, preserving ids, dates,
  numbers and user directives.
- **respond** never writes claims. It first returns the *required* reads with arguments
  resolved by application code (Maya by name/E001, Rachel E010 with status open, laptop
  asset type plus Israel when known, Webex, `equipment_policy` for entitlements); then it
  is offered a **catalogue** of candidate statements and may only return
  `candidate_ids` (+ `unresolved_items`). The application maps ids back to tasks, so
  citations and quotes are never model-authored; invented ids fail closed.

Candidate catalogue: operational rows map record statuses (completed→complete,
pending/open→pending, planned/available→proposed, blocked→blocked; a subscription row is
blocked only when `active_seats ≥ seat_limit`). Documents are split into exact sentences
— chunks are flattened, so recover the first sentence after a `## Heading` by finding
where a sentence starts (a capitalised word followed by a lowercase one, or an opener
like "If"; never drop "If" and turn a conditional into a blocker). A sentence becomes a
task only if `status_is_supported`: proposed ← required, entitlement, need(s), must, may
receive, only after, not permitted/allowed, responsible for, eligible for, verify,
confirm, enroll; pending ← pending, awaiting, require…approval; complete ← completed …
(not under negation or modality); blocked ← blocked, no seats, should not assign, exceed,
but never with may/might/if/can/could, and only if it names something (a system, team or
number). "N active seats against an M-seat limit" is a blocker exactly when N ≥ M.

`validate_tasks` keeps a task only if every citation matches cached evidence exactly,
every support quote is in the cited passage, the description is inside a quote and the
status is supported; otherwise it becomes `unresolved` with a reason. Precedence between
versions of one task: operational record > Maya-specific document > general guidance,
then explicit newer date; equal precedence with different content → `unresolved`
("Conflicting evidence at equal precedence"). Comparable Finance-threshold statements share
one key so the newer one (USD 7,500 for Q3 2026) wins. Application backstops in `respond`:
rows from the turn's required reads are always selected; for a full checklist (below)
every in-scope requirement of `onboarding_policy.md`, `equipment_responsibility_form.md`
and `acceptable_use_policy.md` is selected. The answer prompt says: `unresolved_items` only
for facts no catalogue statement supports — never restate, paraphrase or summarise
catalogue content, facts or prior answers there.

## Planning policy (`policy.py`, `memory.py`)

Intents and read loadouts:

| intent | read tools |
|---|---|
| policy_question | list_policies, get_policy |
| employee_lookup | get_employee |
| onboarding_status | get_employee, list_onboarding_tasks, check_asset_inventory, list_policies, get_policy, check_software_subscription |
| equipment_request | check_asset_inventory, list_policies, get_policy |
| subscription_review | check_software_subscription, get_policy |
| access_request | get_employee, check_software_subscription, list_policies, get_policy |
| ticket_status | get_employee, list_employee_tickets |
| recall, other | none (zero schemas) |

`necessary_reads(plan, text)` names the reads a turn must execute (e.g. ticket_status →
list_employee_tickets; equipment readiness → check_asset_inventory, entitlements →
list_policies/get_policy; checklist → list_onboarding_tasks). The selected loadout is
narrowed to those when no second intent is needed. Application rules override the
planner where the text is unambiguous: explicit source cues (record, tickets, checklist,
laptop, seats/licences, Finance/approval/threshold), recall (`remind me … gave you`,
`full status summary`) → recall with no retrieval or reads, an affirmative
`file/submit/send … Webex … now|request` (no negation, conditional or drafting) →
access_request. Planner facts are kept only if a user said them or memory pinned them
(assistant prose is not a source). Pin from user text before any model call: start date
(`start`, `starts`, `starting`, `start date`; invalid dates are not pinned), Israel,
hybrid, the Q3 SaaS freeze. "drop it / back to Maya" after a Rachel turn closes the
Rachel tangent: remove her evidence, tasks and history from later context; only the
application may close topics (a distilled memory cannot add `closed_topics`).

Distill when the fresh tail has ≥ 12 messages or ≈ 2,500 tokens, folding whole older turns
and keeping the newest two user turns raw; accept a fold only if it keeps every previous
fact/constraint value, decision, unresolved item and closed topic, covers every user
directive in the folded block, and stays under 8,000 characters — otherwise keep raw
history and delay the retry. The conversation history stores a compact view of each
answer (`{status, tasks: [[description, status]], pending_access}`), never the full
response JSON.

## The graph (`graph.py`)

```text
plan → retrieve → select → model ⇄ tools (≤ 2 read rounds, one read-only rearm) → finalize → handoff → END
```

`RECURSION_LIMIT = 12`. Input is exactly `{"newest_message": str}`; the host supplies
`configurable.caller` and a stable `thread_id`; the same thread used by a different caller
is denied; turns of one thread are serialized. If a turn plans reads but executes none,
rearm once with the full read-only catalogue (never a write). `finalize` validates tasks,
adds `unresolved` items for missing required sources, errors or skipped reads, builds the
checklist, and, for an explicit supported Webex request, checkpoints a dispatch record
**before** the `handoff` node calls the injected `WebexPort`. A checkpointer keyed on
thread state plus an in-process guard make replay of turn 10 return the saved
`PendingAccessResult` without calling the port again; an unacknowledged attempt is
reported unresolved and never retried automatically. The idempotency key is
`sha256({version, thread_id, caller, employee E001, system Webex, business_reason})`.
Webex completion is always rejected; responses say pending.

**Full checklist request** — a message containing "checklist" that names two or more of
systems / equipment / policy|acknowledgements / blocked: application-owned plan
(`onboarding_status`, retrieval of `maya_cohen_offer_letter.md,
maya_cohen_onboarding_summary.md, onboarding_policy.md, access_management_policy.md,
salesforce_access_request.md, equipment_policy.md, laptop_provisioning.md,
equipment_responsibility_form.md, acceptable_use_policy.md, webex_license_assignment.md,
webex_license_cleanup_memo.md`), reads `get_employee → list_onboarding_tasks →
check_software_subscription` (the subscription name comes from blocked `Access` onboarding
rows, `Request <System> …`; skipped if none are blocked), all retrieved documents in
focus, location from the employee record when the user did not state it, and rules whose
subject is another audience (contractors unless the record says contractor;
privileged/admin/AWS admin unless asked) removed from the catalogue. The prompt adds:
cover each requested section with statements that apply to this employee; database rows
alone are not a checklist; no rules for other audiences.

## Commands

Every runnable file starts with a guard so `python code/<file>.py` re-runs itself as the
package module (`runpy`), and prints *"maya is not installed for this Python …"* instead
of a traceback under an interpreter without Maya. Every tool answers `--help` without
starting anything. Shortcuts in `pyproject.toml`:

| command | module | does |
|---|---|---|
| `maya-live` | `maya.graph:main` | `--session S2 [--trace]`, a question, or `--chat`; `--as ID/GROUP`, `--thread`; always the live Bedrock model with local documents; prints `model: Bedrock <id>`, per turn `steps=n/12 tokens=in/out time=…s`, and `time: N turn(s) Xs (mean, slowest) + startup + other = total` measured from when `graph.py` starts |
| `maya-eval` | `maya.evals.runner:main` | graded offline replay (rule-based stand-in model, real corpus and records) |
| `maya-ingest` | `maya.ingest:main` | validate the dataset; `--create-index --ingest [--force]` |
| `maya-server` | `maya.server.server:main` | the read-only MCP server |

Also `code/evals/live_model.py [--output]` (graded, live model) and `code/evals/live.py
[--probe] [--prepare]` (graded, Bedrock + OpenSearch + stdio MCP; requires exactly the
seven read tools). `evals/metering.py` wraps the Bedrock client to record tokens and
latency per call site (planner/answer/distill/rerank/embedding) into each turn.
`setup.sh` creates `.venv`, installs, runs tests if `code/tests/` exists, validates the
data, and is safe to `source` from bash or zsh (it re-runs itself in bash, then activates
`.venv`; its `set -euo pipefail` never leaks into the user's shell). `requirements.txt`
pins the tested environment and ends with `-e .`.

## Evaluation (`evals/`)

Adapt a sequential runner: all 12 turns on one thread, then replay turn 10 on the same
thread, then an unauthorized `E002/UG_REGULAR` caller asking for Maya's offer letter, then
Sara's full-checklist request on its own thread. Record per turn: plan, loadout, reads with
arguments and results, retrieval rounds and sources, model calls, handoffs, checklist,
node visits, errors. Deterministic checks per turn: expected/forbidden reads, read
necessity, read-only catalogue, calls reachable, argument checks, required facts (graded
from displayed fields and task text only), closed tangent (no Rachel/T001/E010 after
turn 5), task citations (exact, cached, quote-supported), structured checklist, runtime
errors, recursion bound, handoff count/events/fields, pending acknowledgement, Webex never
granted (turn ≥ 8), and on turn 12 the pending request, date + Israel/hybrid + Q3 freeze,
and the USD 7,500 Finance threshold. Session checks: twelve turns, one thread, every turn
passed, read-only policy, one handoff at turn 10 and none on replay (same request id, one
port call), safe denial with zero dependency and backend calls and no sensitive output,
the first-request checklist (identity fields, all eight homework sources retrieved,
systems/equipment/acknowledgements/blocker cited, blocker cites 42 vs 40, nothing
unresolved), and the tool matrix for all twelve turns (`matrix.py`). A checker fault
fails its turn instead of aborting the run; the report is always written.

## Acceptance

- Offline: all 12 turns, replay, permission case and first request pass (171/171 S2 checks).
- Live (Bedrock + OpenSearch + MCP): 12/12 in repeated runs; the single-turn checklist has
  role, 2026-08-01, Israel, the CSM system bundle, laptop/monitor/dock/headset, the
  signed-offer rule, equipment responsibilities, acceptable use, manager approval, and the
  Webex blocker cited to the subscription record and memo (42/40), pending never granted.
- Unauthorized caller: denied before retrieval, nothing leaked.
- Reference measurements of the original build (Nova 2 Lite): ≈ 3–4 s per turn, ≈ 40–45 s
  for S2 with `maya-live`; ≈ 117k input / 7k output tokens per graded session (planner
  35 %, answer 35 %, rerank 22 %, distill 8 %); final-turn input ≈ 11–14k.
- Robustness: impossible dates, prose dates in memory, malformed Bedrock/OpenSearch/MCP
  responses, missing configuration and a failing reranker never crash a turn — they fail
  closed or degrade with a clear message.

<!-- What this prompt does not include: the unit tests (kept outside version control in
     the original), its replay experiments' captured inputs (results/ is regenerated), and
     a real Webex approval workflow (out of scope; tests use a recording fake that returns
     a deterministic pending id). Expect live numbers to move a few percent between runs. -->
