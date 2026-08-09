# Homework — Lesson 6 RAG · A chunking research study

The lab shipped **one** naive chunking strategy: a fixed 250-word window with 50
words of overlap. But chunking is the single biggest lever on naive-RAG quality —
the same corpus, embedded the same way, retrieves very differently depending on
how you cut it up.

**Your job is a small research study: find the best chunking strategy for
*this* corpus** (the NovaOps handbook), judged by retrieval quality against the
cost of building the index. This isn't a "make it work" exercise — it's "measure
several options and defend a winner."

> **Goal / final-project connection.** The strategy and parameters you land on
> become your final project's **retrieval configuration** — the baseline every
> later retrieval improvement is measured against. You're not tuning a toy; you're
> choosing the default your project assistant will ship with, and producing the
> fixed question set you'll keep re-testing it with.

## The techniques to try

Build a few indexes, each with a different strategy (and combine them where it
makes sense — *overlapping* in particular is a modifier you can layer on any of
the others):

| Technique            | How it works                                          | When to use                        |
| -------------------- | ----------------------------------------------------- | ---------------------------------- |
| Fixed-size           | Split every 300–800 tokens                            | Simple demos / baseline            |
| Separator-based      | Split on `\n\n`, headings, bullets, sentence periods  | Docs with clear structure          |
| Sentence-based       | 3–8 sentences per chunk                               | Explanatory text                   |
| Overlapping          | Repeat 50–200 tokens between chunks                   | Avoid losing meaning at boundaries |
| Semantic / LLM-based | Split when the topic/context changes                  | Policies, legal, research          |

The handbook is **markdown with real structure** (headings, bullets, short
sections) — so the separator- and sentence-based strategies have something to bite
on, and it's a fair fight against the fixed-size baseline. Keep the lab's fixed
250/50 index as your **control** to compare everything against.

> Tokens vs. words: the lab chunks by *words*; the table is in *tokens*. Don't
> overthink it — a word ≈ 1.3 tokens is fine for planning, or use a real
> tokenizer if you want the counts exact.

**Store every index the same way — with a source tag.** Whatever strategy cuts
the chunks, record each chunk's **source filename** in the index metadata, right
next to its text. The lab kept only the chunk text in `metadata.json`, so this is
a real (small) change you make in ingest — carry the file each chunk came from
into what you store. Do it the same way for every strategy, and you'll lean on it
**twice**: to score retrieval objectively (did the top-k come from the *right*
file?) and to make the assistant **cite its sources**. Build it in from the first
index, not as an afterthought.

## What to record — per index

1. **Build cost — call counts.** How many calls to (a) the **embedding model**
   (Titan) and (b) the **text model** (Nova 2 Lite) it took to build the index.
   Most strategies call only the embedder — **one call per chunk**. The
   semantic/LLM-based strategy *also* calls Nova to decide where to split, so
   watch that number appear (and jump).
   - **1b. What did each index cost?** Turn the calls into dollars: embedding
     tokens × Titan price + Nova input/output tokens × Nova price. Pull the token
     counts from each response (`usage` on Nova; the input length for Titan) and
     look up current Bedrock pricing. This is the number that makes "LLM-based
     chunking is *expensive* to build" concrete rather than hand-wavy.
2. **A test set — 10+ questions.** Write **at least 10** handbook questions you
   run against *every* index unchanged, so the comparison is fair. Mix them:
   several single-fact lookups (vacation, moonlighting, returning your laptop,
   how promotions work), a couple that span two sections of one document, and 1–2
   the handbook genuinely **can't** answer (to check grounding/refusal). This set
   is a deliverable — you keep reusing it.
3. **Retrieval quality — per question, per index.** For each question, record the
   retrieved chunks — ids, **source files**, and similarity scores. Because every
   chunk is source-tagged, you can score this **objectively**: note upfront which
   handbook file actually answers each test question, then measure the
   **right-source hit rate** — did the top-k include a chunk from that file? That's
   a far cleaner cross-strategy number than eyeballing text or cosines.
   - ⚠️ **Don't rank strategies by raw cosine.** Smaller chunks inflate the
     absolute score (you saw why in the lab — a short chunk dilutes the query
     less), so a strategy can look "better" on score while retrieving worse. Rank
     on the right-source hit rate and whether `rag_tool.py` gives the **correct,
     grounded answer**; treat the scores as a secondary signal.
4. **Optional — pre/post-processing.** You may clean the raw text before chunking
   (strip nav/boilerplate, normalize headings) or fix up chunks afterward (drop
   tiny fragments, merge orphan lines, dedupe). Not required — but if you try it,
   note whether it moved the needle.

## How to test — reference `rag_tool.py`

Point the lab's tool-based RAG (`02-rag-as-tool/rag_tool.py`) at each index in
turn and compare the **answers**, not just the scores. The winner is the strategy
whose answers are most consistently correct and grounded across your 10+ questions
— at a build cost you'd actually be willing to pay.

- **Optional:** repeat the study on the `manager_playbook` corpus. Keep each
  index **single-corpus** — you do *not* need to make one query search both the
  handbook and the playbook here. (Cross-source questions are a separate problem;
  leave them for now.)

## ⭐ Cite every answer's source — treat this as a headline task, not a footnote

> **Make every answer name the document each claim came from** — e.g. *"outside
> work is fine with prior approval **(moonlighting.md)**"*. When you build each
> index you tag every chunk with its source file (the source tag above) — here you
> carry that tag the last step, into the generated answer. Concretely: tag each
> retrieved chunk with its file when you build the context, and add a system-prompt
> rule that **every factual claim must cite the source file it came from**. This is the single piece
> of this work that ships almost unchanged into your final project — build it
> properly.

**Why citations matter — in any real RAG system:**

- **Trust & verifiability.** The reader can check the claim against the real doc.
- **Contains hallucination.** Uncited or mis-cited claims stand out.
- **Auditability & compliance.** The citation *is* the audit trail.
- **Debuggability.** Pinpoints failures — retrieval vs. generation.

**Why citations matter — specifically for your benchmark:**

- **They make the retrieval decision visible in the output.** The cited file in
  each answer *is* the right-source signal (item 3) surfaced automatically — you
  read the answer and immediately see whether the strategy leaned on the correct
  document, instead of hand-checking chunk text.
- **They separate the two failure modes per strategy** — exactly the comparison
  you're running. A strategy that cites the right file but answers poorly is a
  *retrieval win / generation issue*; one that cites the wrong file is a *chunking
  loss*. Without citations these look identical, and you'd mis-rank the strategies.
- **They keep the study honest.** Because the corpus is adapted from a public
  handbook, the model can sometimes answer from memory without really using
  retrieval. A citation requirement forces every claim back onto the retrieved
  chunks — so you're benchmarking *retrieval*, not the model's prior knowledge.

**Project piece (M2 — cite-your-policy):** this is a hard requirement in the final
project spec. Build the mechanism once, here, and reuse it unchanged.

## Deliverable

A short **comparison table** — one row per index:

| strategy (params) | # chunks | embed calls | Nova calls | ~$ to build | right-source hits (of N) | answers correct |
| ----------------- | -------- | ----------- | ---------- | ----------- | ------------------------ | --------------- |

— plus a **one-paragraph recommendation**: the sweet spot for this corpus, and the
tradeoff you made to pick it (e.g. "separator + 100-token overlap: nearly the
retrieval quality of the LLM-chunked index at a fraction of the build cost").

## Starter prompt

> "Create a `chunking_study/` harness (importing the lab's embed/search helpers)
> that builds several FAISS indexes over `data/handbook/`, each in its own
> `data/faiss_index_*` dir so runs don't clobber each other: (1) fixed-size at
> 300/600/800 words with 50–200-word overlap, (2) separator-based on markdown
> headings and paragraphs, (3) sentence-based at 3–8 sentences per chunk, (4)
> semantic/LLM-based that asks Nova 2 Lite where the topic changes and splits
> there. Record each chunk's **source filename** in the metadata, and update
> `rag_tool.py` to tag context chunks with their source and cite the source file
> in every answer. For each index, count and print the **embedding-model calls**
> and **Nova calls** used to build it, and estimate the **dollar cost** from the
> token counts (`usage` for Nova; input length for Titan) and current Bedrock
> pricing. Then run this fixed list of questions [I'll paste 10+] against every
> index through `rag_tool.py`, printing per index the retrieved chunk ids +
> cosine scores + source files and the final cited answer. I'll also give an
> expected source file per question — score the **right-source hit rate** (did the
> top-k include a chunk from that file?) and emit a comparison table (strategy ·
> #chunks · embed calls · Nova calls · $ · right-source hits · answers correct)."

Iterate from there: tighten the winning strategy's parameters, add your optional
pre/post-processing, and settle on the config you'd ship.

## What the project takes from this work

This is the first real piece of your **final project's** retrieval layer. Three
things you build here carry straight in: the **default chunking config** the
assistant ships with (chosen by measurement), the **source-citation mechanism**
(M2 — a hard requirement), and the **fixed question set + cost baseline** you'll
keep re-testing every future retrieval change against.
