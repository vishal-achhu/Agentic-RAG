import json

import pytest
from fastapi.testclient import TestClient

from agentic_rag.api import create_app
from agentic_rag.evaluation import eval_end_to_end, eval_retrieval, load_eval
from conftest import EVAL


@pytest.fixture()
def client(svc):
    return TestClient(create_app(svc))


def test_health_query_metrics(client):
    assert client.get("/health").json()["chunks"] > 0
    r = client.post("/query", json={"question": "What is the free trial length?"}).json()
    assert "14 day" in r["answer"] and r["citations"] and r["grounding"] == 1.0
    assert "rag_requests_total 1" in client.get("/metrics").text
    assert client.post("/query", json={"question": "x"}).status_code == 422


def test_ingest_endpoint_then_query(client):
    r = client.post("/ingest", json={"documents": [{"doc_id": "faq", "text": "# FAQ\n\n## Mascot\nThe Nimbus mascot is a friendly otter named Ollie."}]})
    assert r.json()["added"] == 1
    assert "otter" in client.post("/query", json={"question": "What is the Nimbus mascot?"}).json()["answer"]


def test_sse_stream_emits_steps_tokens_final(client):
    body = client.get("/query/stream", params={"q": "What is the free trial length?"}).text
    events = [l.split(": ", 1)[1] for l in body.splitlines() if l.startswith("event: ")]
    assert events[0] == "step" and "token" in events and events[-1] == "final"
    final = json.loads([l for l in body.splitlines() if l.startswith("data: ")][-1][6:])
    assert final["result"]["citations"]


def test_api_key_enforced(svc, monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "s3cret")
    c = TestClient(create_app(svc))
    assert c.post("/query", json={"question": "free trial?"}).status_code == 401
    assert c.post("/query", json={"question": "free trial?"}, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert c.get("/health").status_code == 200


def test_quality_gate_on_golden_set(svc):
    """CI regression gate: fails the build if retrieval or agent quality drops."""
    rows = load_eval(EVAL)
    ret = eval_retrieval(svc, rows)
    assert ret["hybrid (RRF)"]["hit"] >= 0.95
    assert ret["hybrid (RRF)"]["mrr"] >= ret["bm25"]["mrr"]       # fusion must not be worse than lexical alone
    e2e = eval_end_to_end(svc, rows)
    assert e2e["unanswerable_pass_rate"] == 1.0 and e2e["attack_pass_rate"] == 1.0
    assert e2e["answerable_pass_rate"] >= 0.7 and e2e["mean_grounding"] >= 0.9
