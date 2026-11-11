import json
import os
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from .service import RAGService


class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class DocIn(BaseModel):
    doc_id: str = Field(min_length=1, max_length=128, pattern=r"^[\w.\-]+$")
    text: str = Field(min_length=1, max_length=500_000)
    title: Optional[str] = None


class IngestIn(BaseModel):
    documents: list[DocIn] = Field(min_length=1, max_length=100)


def create_app(service: Optional[RAGService] = None) -> FastAPI:
    svc = service or RAGService()
    app = FastAPI(title="Agentic RAG API", version="1.0.0")
    required_key = os.getenv("RAG_API_KEY")

    def auth(x_api_key: Optional[str] = Header(default=None)) -> None:
        if required_key and x_api_key != required_key:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": len(svc.store), "llm": svc.llm.name, "embedder": svc.embedder.name}

    @app.post("/ingest", dependencies=[Depends(auth)])
    def ingest(body: IngestIn):
        return svc.ingest_texts([d.model_dump() for d in body.documents])

    @app.post("/query", dependencies=[Depends(auth)])
    def query(body: QueryIn):
        return svc.ask(body.question).to_dict()

    @app.get("/query/stream", dependencies=[Depends(auth)])
    def query_stream(q: str):
        if not 3 <= len(q) <= 2000:
            raise HTTPException(status_code=422, detail="q must be 3-2000 characters")

        def gen():
            for ev in svc.ask_stream(q):
                yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return svc.metrics.render()

    return app


def app_factory() -> FastAPI:          # uvicorn agentic_rag.api:app_factory --factory
    return create_app()
