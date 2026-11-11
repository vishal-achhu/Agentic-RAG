from __future__ import annotations

import argparse
import json
import sys

from .config import Settings
from .evaluation import eval_end_to_end, eval_retrieval, format_report, load_eval
from .service import RAGService


def _ensure_index(svc: RAGService, docs: str) -> None:
    if len(svc.store) == 0:
        print(f"[index empty: ingesting {docs}]", file=sys.stderr)
        svc.ingest_dir(docs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentic-rag", description="Agentic RAG toolkit")
    p.add_argument("--docs", default="data/docs")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--json", action="store_true")
    a.add_argument("--stream", action="store_true")
    sub.add_parser("chat")
    e = sub.add_parser("eval")
    e.add_argument("--file", default="data/eval/golden.jsonl")
    e.add_argument("--json", action="store_true")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    svc = RAGService(Settings())
    if args.cmd == "ingest":
        print(json.dumps(svc.ingest_dir(args.docs), indent=2))
        return 0
    _ensure_index(svc, args.docs)

    if args.cmd == "ask":
        if args.stream:
            for ev in svc.ask_stream(args.question):
                if ev["type"] == "step":
                    st = ev["step"]
                    print(f"  step {st['index']}: {st['tool']}({st['input']!r}) -> {st['observation']}", file=sys.stderr)
                elif ev["type"] == "token":
                    print(ev["text"], end="", flush=True)
            print()
        else:
            r = svc.ask(args.question)
            print(json.dumps(r.to_dict(), indent=2) if args.json else _pretty(r))
    elif args.cmd == "chat":
        print("Ask anything about the knowledge base (Ctrl-D to quit).")
        for line in sys.stdin:
            if line.strip():
                print(_pretty(svc.ask(line.strip())), "\n")
    elif args.cmd == "eval":
        rows = load_eval(args.file)
        retrieval, e2e = eval_retrieval(svc, rows), eval_end_to_end(svc, rows)
        print(json.dumps({"retrieval": retrieval, "end_to_end": e2e}, indent=2) if args.json
              else format_report(retrieval, e2e))
    elif args.cmd == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(svc), host=args.host, port=args.port)
    return 0


def _pretty(r) -> str:
    out = [f"Q: {r.question}", f"A: {r.answer}"]
    if r.citations:
        out.append("Sources: " + "; ".join(f"[{c['n']}] {c['title']} > {c['section'] or 'intro'}" for c in r.citations))
    out.append(f"(grounding={r.grounding} relevance={r.top_relevance} steps={len(r.steps)} "
               f"{r.latency_ms}ms{' CACHED' if r.cached else ''}{' ' + str(r.warnings) if r.warnings else ''})")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
