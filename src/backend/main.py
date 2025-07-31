from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.requests import Request
from starlette.routing import Route
from starlette.staticfiles import StaticFiles
from pathlib import Path

# --- Database integration ---
from src.database.data import DatabaseAPI, validate_db_dict
from src.database.db_init import build_initial_db_dict

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend2"

# Build and validate the initial db dict
db_dict = build_initial_db_dict()
validate_db_dict(db_dict)

# Initialize the database
db = DatabaseAPI()
db.set_samples([s["filepath"] for s in db_dict["samples"]])

# In-memory accuracy stats
accuracy_stats = {"tries": 0, "correct": 0}


def create_image_response(image_path):
    """Helper function to create image response with headers"""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    
    # Get label or prediction data
    anns = db.get_annotations(image_path)
    label_ann = next((a for a in anns if a.get('type') == 'label'), None)
    
    headers = {
        "X-Image-Id": str(image_path)  # Always include the image ID
    }
    if label_ann:
        headers["X-Label-Class"] = str(label_ann.get('class', ''))
        headers["X-Label-Source"] = "annotation"
    else:
        # Look for prediction
        preds = db.get_predictions(image_path)
        pred_ann = next((p for p in preds if p.get('type') == 'label' and p.get('probability') is not None), None)
        if pred_ann:
            headers["X-Label-Class"] = str(pred_ann.get('class', ''))
            headers["X-Label-Source"] = "prediction"
            headers["X-Label-Probability"] = str(pred_ann.get('probability', ''))
    
    return FileResponse(image_path, media_type=mime_type, headers=headers)


async def get_sample_by_id(request):
    # Get id from query parameter, must be in the database
    id_param = request.query_params.get("id")
    all_images = db.get_samples()
    if id_param not in all_images:
        # this check is important, without it the client can access any file in the system
        raise ValueError(f"Image with id '{id_param}' not found.")
    return create_image_response(id_param)



async def get_next_sample(request):
    # Find the next unlabeled image, skipping the current one
    current_id = request.query_params.get("current_id")
    all_images = db.get_samples()
    for image_path in all_images:
        if image_path == current_id:
            continue
        anns = db.get_annotations(image_path)
        label_ann = next((a for a in anns if a.get('type') == 'label'), None)
        if not label_ann:  # No label annotation found, this is unlabeled
            return create_image_response(image_path)
    # No unlabeled images found
    return JSONResponse({"error": "No unlabeled images available"}, status_code=404)


async def handle_annotation(request: Request):
    if request.method == "POST":
        data = await request.json()
        filepath = data.get("filepath")
        class_name = data.get("class")
        if not filepath or not isinstance(class_name, str):
            return JSONResponse({"error": "Missing or invalid 'filepath' or 'class'"}, status_code=400)
        # Write annotation to DB
        db.save_label_annotation(filepath, class_name)

        # Accuracy tracking: check prediction
        preds = db.get_predictions(filepath)
        pred_ann = next((p for p in preds if p.get('type') == 'label' and p.get('probability') is not None), None)
        if pred_ann:
            accuracy_stats["tries"] += 1
            if str(pred_ann.get("class")) == str(class_name):
                accuracy_stats["correct"] += 1
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

async def get_accuracy_stats(request: Request):
    tries = accuracy_stats["tries"]
    correct = accuracy_stats["correct"]
    accuracy = (correct / tries) if tries > 0 else None
    return JSONResponse({
        "tries": tries,
        "correct": correct,
        "accuracy": accuracy
    })

config = dict()

async def put_config(request: Request):
    data = await request.json()
    if not isinstance(data, dict):
        return JSONResponse({"error": "Invalid config format"}, status_code=400)
    
    # Validate and save the config
    config.update(data)
    print("Config updated:", config)
    return JSONResponse({"status": "Config saved successfully"})

async def get_config(request: Request):
    if not config:
        return JSONResponse({"error": "No config set"}, status_code=404)
    
    print("Config updated", config)
    return JSONResponse(config)

async def get_stats(request: Request):
    # Return the accuracy stats
    return JSONResponse(accuracy_stats)

app = Starlette(
    routes=[
        Route("/config", put_config, methods=["PUT"]),
        Route("/config", get_config, methods=["GET"]),
        Route("/next", get_next_sample, methods=["GET"]),
        Route("/sample", get_sample_by_id, methods=["GET"]),
        Route("/annotate", handle_annotation, methods=["POST"]),
        Route("/annotate", handle_annotation, methods=["DELETE"]),
        Route("/stats", get_stats, methods=["GET"]),
    ]
)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
