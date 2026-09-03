"""The fixed 12-question test set every index is scored against (README's
"a test set — 10+ questions"). Mixes single-fact lookups, questions that span
two sections of one document, and unanswerable questions that check the
assistant refuses instead of hallucinating.

`expected_source` is None for the unanswerable questions — there's no right
file to hit, so those two are excluded from the right-source hit-rate and
instead scored on whether the model refused.

`keywords` are the facts a CORRECT answer must contain, used as an automatic,
reproducible proxy for "answer correct" in place of a human reading every
cell of a 12-question x 8-index grid. Any one keyword group (an inner list)
matching is enough — some facts can be phrased more than one way.
"""

QUESTIONS = [
    {
        "question": "How many paid vacation days do full-time employees get per year?",
        "expected_source": "benefits-and-perks.md",
        "keywords": [["20"]],
    },
    {
        "question": "What is NovaOps's policy on employees doing outside paid work (moonlighting)?",
        "expected_source": "moonlighting.md",
        "keywords": [["conflict"], ["disclose"], ["competitor"]],
    },
    {
        "question": "What should I do with my laptop when I leave the company?",
        "expected_source": "managing-work-devices.md",
        "keywords": [["five business days"], ["return"]],
    },
    {
        "question": "What severance do employees get if terminated without cause?",
        "expected_source": "severance.md",
        "keywords": [["4 weeks"], ["2 weeks"], ["16 weeks"]],
    },
    {
        "question": "How often do performance reviews happen and where are they recorded?",
        "expected_source": "making-a-career.md",
        "keywords": [["February"], ["August"], ["BambooHR"]],
    },
    {
        "question": "What are the title levels for QA engineers?",
        "expected_source": "titles-for-QA.md",
        "keywords": [["QA1"], ["QA2"], ["QA3"]],
    },
    {
        "question": "What tools/systems does NovaOps use for HR records and access?",
        "expected_source": "our-internal-systems.md",
        "keywords": [["BambooHR"], ["Okta"]],
    },
    {
        "question": "What recurring rituals/meetings does NovaOps run, such as standups?",
        "expected_source": "our-rituals.md",
        "keywords": [["weekly company update"], ["standup"]],
    },
    {
        "question": (
            "What's the difference between the quarterly growth conversation, the formal "
            "performance review, and the promotion cycle — how often does each happen?"
        ),
        "expected_source": "making-a-career.md",
        "keywords": [["quarterly"], ["February", "August"], ["March", "September"]],
    },
    {
        "question": "Who is eligible for NovaOps benefits, and how does that interact with how vacation days are earned?",
        "expected_source": "benefits-and-perks.md",
        "keywords": [["full-time"], ["20"], ["rollover", "rolls over", "capped"]],
    },
    {
        "question": "What is NovaOps's current stock price?",
        "expected_source": None,
        "keywords": [],  # correctness = refusal, not a fact match — see is_refusal()
    },
    {
        "question": "What is the CEO's personal cell phone number?",
        "expected_source": None,
        "keywords": [],
    },
]

_REFUSAL_PHRASES = [
    "don't have", "do not have", "doesn't contain", "does not contain",
    "don't know", "do not know", "not contain", "no information",
    "can't find", "cannot find", "couldn't find", "not available in",
    "not something i can", "doesn't include", "does not include",
    "can't share", "cannot share", "can't provide", "cannot provide",
    "i can only", "i can't help", "i cannot help", "not something i have",
]


def is_refusal(answer: str) -> bool:
    """Heuristic check for whether an answer declined to invent an unsupported
    fact, used to grade the two unanswerable questions."""
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)


def is_correct(question: dict, answer: str) -> bool:
    """Automatic correctness grade: for unanswerable questions, correct means
    refused; for answerable ones, correct means every keyword group has at
    least one match in the answer (case-insensitive)."""
    if question["expected_source"] is None:
        return is_refusal(answer)
    lowered = answer.lower()
    return all(any(kw.lower() in lowered for kw in group) for group in question["keywords"])
