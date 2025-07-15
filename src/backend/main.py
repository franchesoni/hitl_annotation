from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
import mimetypes
from starlette.routing import Route
from starlette.staticfiles import StaticFiles
from pathlib import Path

# --- Database integration ---
from src.database.data import DatabaseAPI, validate_db_dict
from src.database.db_init import build_initial_db_dict

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# Build and validate the initial db dict
db_dict = build_initial_db_dict()
validate_db_dict(db_dict)

# Initialize the database
db = DatabaseAPI()
db.set_samples([s["filepath"] for s in db_dict["samples"]])
# Optionally, you could also set annotations and predictions here if needed


async def sample(request):
    # Get id from query parameter, must be in the database
    id_param = request.query_params.get("id")
    all_images = db.get_samples()
    if id_param not in all_images:
        raise ValueError(f"Image with id '{id_param}' not found.")
    image_path = id_param
    import mimetypes

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    return FileResponse(image_path, media_type=mime_type)


async def get_ids(request):
    # Return all sample filepaths from the database
    return JSONResponse(db.get_samples())


app = Starlette(
    routes=[
        Route("/api/sample", sample),
        Route("/api/ids", get_ids),
    ]
)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
