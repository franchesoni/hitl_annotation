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
        # this check is important, without it the client can access any file in the system
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




from starlette.requests import Request
async def get_label_annotation(request: Request):
    # Get id from query parameter (filepath)
    filepath = request.query_params.get("filepath")
    if not filepath:
        return JSONResponse({"error": "Missing 'filepath'"}, status_code=400)
    anns = db.get_annotations(filepath)
    # Find the label annotation (type == 'label')
    label_ann = next((a for a in anns if a.get('type') == 'label'), None)
    if label_ann:
        return JSONResponse({"class": label_ann.get('class')})
    else:
        return JSONResponse({"class": None})

async def save_label(request: Request):
    if request.method == "POST":
        data = await request.json()
        filepath = data.get("filepath")
        class_name = data.get("class")
        if not filepath or not isinstance(class_name, str):
            return JSONResponse({"error": "Missing or invalid 'filepath' or 'class'"}, status_code=400)
        # Write annotation to DB
        db.save_label_annotation(filepath, class_name)
        return JSONResponse({"status": "ok"})
    elif request.method == "DELETE":
        data = await request.json()
        filepath = data.get("filepath")
        if not filepath:
            return JSONResponse({"error": "Missing 'filepath'"}, status_code=400)
        db.delete_label_annotation(filepath)
        return JSONResponse({"status": "deleted"})
    else:
        return JSONResponse({"error": "Method not allowed"}, status_code=405)

app = Starlette(
    routes=[
        Route("/api/sample", sample),
        Route("/api/ids", get_ids),
        Route("/api/save_label", save_label, methods=["POST", "DELETE"]),
        Route("/api/get_label_annotation", get_label_annotation, methods=["GET"]),
    ]
)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
