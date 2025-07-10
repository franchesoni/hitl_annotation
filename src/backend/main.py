from starlette.applications import Starlette
from starlette.responses import FileResponse
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
IMAGE_PATH = IMAGES[1] 

async def sample(request):
    mime_type, _ = mimetypes.guess_type(IMAGE_PATH)
    if mime_type is None:
        mime_type = "application/octet-stream"
    return FileResponse(IMAGE_PATH, media_type=mime_type)

app = Starlette(routes=[Route("/api/sample", sample)])
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
