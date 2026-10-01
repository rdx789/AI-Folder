"""Replay the real Bedrock model against fixed evidence and fake Webex."""
if __name__ == '__main__' and not __package__:
    # Started by file path (python code/evals/live_model.py): relative imports need the package, so
    # re-run this file as the installed module `maya.evals.live_model`.
    import runpy
    try:
        runpy.run_module('maya.evals.live_model', run_name='__main__', alter_sys=True)
    except ImportError as exc:  # runpy wraps the ModuleNotFoundError for the package
        if 'maya' not in (getattr(exc, 'name', None), getattr(exc.__cause__, 'name', None)):
            raise
        raise SystemExit("maya is not installed for this Python. Use the project's interpreter: "
                         ".venv/bin/python (or `source .venv/bin/activate`), or run ./setup.sh")
    raise SystemExit
import asyncio
import json
from pathlib import Path

from ..graph import Dependencies
from ..paths import DATA_PATH, RESULTS_DIR
from ..ingestion import load_chunks
from ..model import live_model
from ..operations import LocalReadPort
from ..ports import FakeWebexPort
from ..retrieval import OpenSearchEvidenceRetriever
from .capture import build_recorded_graph
from .metering import MeteredBedrock, print_summary, summarize, write_requests
from .offline import OfflineRuntime, OfflineSearch
from .runner import load_session, run_evaluation


async def main(output=None):
    search = OfflineSearch(load_chunks(DATA_PATH))
    retriever = OpenSearchEvidenceRetriever(
        search, index='offline', embed=search.embed, rerank=search.rerank)
    webex = FakeWebexPort()
    model = live_model()
    model.client = MeteredBedrock(model.client)
    app, capture = build_recorded_graph(
        Dependencies(model, retriever, webex, LocalReadPort()))
    model.client.sink = capture.llm
    report = await run_evaluation(OfflineRuntime(app, capture, webex, search), load_session())
    report['mode'] = 'live-model'
    report['model_id'] = model.model_id
    report['quality_scope'] = 'Live Bedrock model; offline retrieval; local reads; fake Webex.'
    report['token_summary'] = summarize([*report['records'], report['replay'], report['first_request']])
    print_summary(report['token_summary'])
    output = Path(output) if output else RESULTS_DIR / 'latest-s2-live-model.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    write_requests(model.client.requests, output)
    output.write_text(json.dumps(report, indent=2, default=str))
    print(f'Report: {output}')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    import argparse
    from ..backend import external_errors
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='report path (default results/latest-s2-live-model.json)')
    args = parser.parse_args()  # --help and bad arguments stop here, before any paid call
    try:
        raise SystemExit(asyncio.run(main(args.output)))
    except external_errors() as exc:
        raise SystemExit(f'Live-model replay failed: {type(exc).__name__}: {exc}')
