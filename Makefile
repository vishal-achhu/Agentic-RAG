.PHONY: install test eval serve docker
install:; pip install -r requirements.txt
test:;    python -m pytest -q
eval:;    python -m agentic_rag eval
serve:;   python -m agentic_rag serve
docker:;  docker build -t agentic-rag . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY agentic-rag
