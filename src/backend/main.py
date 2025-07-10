from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
import mimetypes
from starlette.routing import Route
from starlette.staticfiles import StaticFiles
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
IMAGES = [
    "/home/franchesoni/mine/repos/annotation_apps/bbox_class_editor/data/assembled_image_img_1_1609.125_1817.0.png",
    "/home/franchesoni/mine/repos/annotation_apps/bbox_class_editor/data/assembled_image_img_1_2699.125_464.0.png",
]


async def sample(request):
    # Get id from query parameter, default to first image if not provided or not found
    id_param = request.query_params.get("id")
    if id_param not in IMAGES:
        raise ValueError(f"Image with id '{id_param}' not found.")
    image_path = id_param
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    return FileResponse(image_path, media_type=mime_type)


async def get_ids(request):
    return JSONResponse(IMAGES)


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
