"""Caller-safe replacement for the lesson's unfiltered FAISS document searches."""
from ..backend import external_errors, live_retriever
from ..retrieval import retrieve_evidence_node


async def search(plan, config, *, retriever=None):
    """Application-only API; authenticated caller is supplied in runtime config."""
    # Authorize before constructing AWS clients or resolving credentials.
    from ..access import caller_from_config, authorize_subject, PermissionDenied, DENIAL
    from ..retrieval import RetrievalResult
    try:
        caller = caller_from_config(config)
        authorize_subject(caller, plan.subject_employee_id)
    except PermissionDenied:
        return RetrievalResult('denied', message=DENIAL)
    if not plan.requires_retrieval:
        return RetrievalResult('skipped')
    try:
        return await retrieve_evidence_node(plan, config, retriever or live_retriever())
    except external_errors() as exc:
        # Backend/config outage: unresolved, no evidence, never a guessed answer.
        return RetrievalResult('unresolved', message=f'Evidence retrieval failed: {type(exc).__name__}')
