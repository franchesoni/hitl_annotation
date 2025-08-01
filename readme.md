# The ultimate image annotation tool

Fast image labeling with human-in-the-loop training. The Starlette backend serves a simple JS frontend and a FastAI loop keeps learning from your annotations. Model checkpoints are saved next to the dataset so progress persists across training runs.

Run the server with:

```bash
uvicorn src.backend.main:app
```

See [SPEC.md](SPEC.md) for full API details and the project TODO list.
