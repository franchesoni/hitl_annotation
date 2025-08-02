from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.requests import Request
from starlette.routing import Route
from starlette.staticfiles import StaticFiles
from starlette.background import BackgroundTask
import mimetypes
import os
import tempfile
from pathlib import Path
from collections import defaultdict, deque

# --- Database integration ---
from src.database.data import DatabaseAPI, validate_db_dict
from src.database.db_init import build_initial_db_dict

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend2"

# Initialize the database
db = DatabaseAPI()
if not db.get_samples():
    db_dict = build_initial_db_dict()
    validate_db_dict(db_dict)
    db.set_samples([s["filepath"] for s in db_dict["samples"]])
config = db.get_config() or {}

# Track the last image served to the client
last_image_served = None

# Buffer storing recent annotations (filepath, class)
ANNOTATION_BUFFER_SIZE = 20
annotation_buffer = deque(maxlen=ANNOTATION_BUFFER_SIZE)


def create_image_response(image_path):
    """Helper function to create image response with headers"""
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
    
    global last_image_served
    last_image_served = str(image_path)

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
    current_id = request.query_params.get("current_id")
    strategy = request.query_params.get("strategy", "least_confident_minority")

    # Sequential strategy (existing behaviour)
    def sequential_next():
        filepath = db.get_next_unlabeled_sequential(current_id)
        if filepath:
            return create_image_response(filepath)
        return JSONResponse({"error": "No unlabeled images available"}, status_code=404)

    if strategy == "sequential":
        return sequential_next()
    if strategy == "least_confident_minority":
        filepath = db.get_next_unlabeled_default(current_id)
        if filepath:
            return create_image_response(filepath)
        return sequential_next()
    if strategy == "last_class":
        if annotation_buffer:
            target_class = annotation_buffer[-1][1]
            filepath = db.get_next_unlabeled_for_class(target_class, current_id)
            if filepath:
                return create_image_response(filepath)
        # Fallbacks if no sample found for last class
        filepath = db.get_next_unlabeled_default(current_id)
        if filepath:
            return create_image_response(filepath)
        return sequential_next()
    if strategy == "specific_class":
        target_class = request.query_params.get("class")
        if target_class:
            filepath = db.get_next_unlabeled_for_class(target_class, current_id)
            if filepath:
                return create_image_response(filepath)
        filepath = db.get_next_unlabeled_default(current_id)
        if filepath:
            return create_image_response(filepath)
        return sequential_next()
    return JSONResponse({"error": "Invalid strategy"}, status_code=400)


async def handle_annotation(request: Request):
    global annotation_buffer
    if request.method == "POST":
        data = await request.json()
        filepath = data.get("filepath")
        class_name = data.get("class")
        if not filepath or not isinstance(class_name, str):
            return JSONResponse({"error": "Missing or invalid 'filepath' or 'class'"}, status_code=400)
        # Write annotation to DB
        db.save_label_annotation(filepath, class_name)

        # Store in recent annotation buffer
        annotation_buffer.append((filepath, class_name))

        # Accuracy tracking: check prediction
        preds = db.get_predictions(filepath)
        pred_ann = next((p for p in preds if p.get('type') == 'label' and p.get('probability') is not None), None)
        if pred_ann:
            was_correct = str(pred_ann.get("class")) == str(class_name)
            db.increment_accuracy(was_correct)
            db.log_accuracy(was_correct)
        return JSONResponse({"status": "ok"})
    elif request.method == "DELETE":
        data = await request.json()
        filepath = data.get("filepath")
        if not filepath:
            return JSONResponse({"error": "Missing 'filepath'"}, status_code=400)
        db.delete_label_annotation(filepath)
        # Remove any occurrences from the annotation buffer
        annotation_buffer = deque(
            [p for p in annotation_buffer if p[0] != filepath],
            maxlen=ANNOTATION_BUFFER_SIZE,
        )
        return JSONResponse({"status": "deleted"})
    else:
        return JSONResponse({"error": "Method not allowed"}, status_code=405)

async def get_accuracy_stats(request: Request):
    stats = db.get_accuracy_counts()
    tries = stats["tries"]
    correct = stats["correct"]
    accuracy = (correct / tries) if tries > 0 else None
    return JSONResponse({
        "tries": tries,
        "correct": correct,
        "accuracy": accuracy
    })

async def put_config(request: Request):
    data = await request.json()
    if not isinstance(data, dict):
        return JSONResponse({"error": "Invalid config format"}, status_code=400)
    
    # Merge and save the config in the database
    config.update(data)
    db.update_config(config)
    print("Config updated:", config)
    return JSONResponse({"status": "Config saved successfully"})

async def get_config(request: Request):
    cfg = db.get_config()
    if not cfg:
        return JSONResponse({"error": "No config set"}, status_code=404)

    print("Config updated", cfg)
    return JSONResponse(cfg)

async def get_stats(request: Request):
    annotated = db.count_labeled_samples()
    total = db.count_total_samples()
    ann_counts = db.get_annotation_counts()
    pct_param = request.query_params.get("pct")
    try:
        pct = float(pct_param) if pct_param is not None else 100.0
    except ValueError:
        pct = 100.0
    stats = db.get_accuracy_counts() if pct >= 100 else db.get_accuracy_recent_pct(pct)
    tries = stats["tries"]
    correct = stats["correct"]
    accuracy = (correct / tries) if tries > 0 else None
    error = (1 - accuracy) if accuracy is not None else None
    return JSONResponse(
        {
            "image": last_image_served,
            "annotated": annotated,
            "total": total,
            "tries": tries,
            "correct": correct,
            "accuracy": accuracy,
            "error": error,
            "annotation_counts": ann_counts,
        }
    )

async def get_training_stats(request: Request):
    stats = db.get_training_stats()
    return JSONResponse(stats)


async def export_db(request: Request):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.close()
    try:
        db.export_db_as_json(tmp.name)
    except Exception:
        os.unlink(tmp.name)
        raise
    return FileResponse(
        tmp.name,
        media_type="application/json",
        filename="db_export.json",
        background=BackgroundTask(lambda: os.remove(tmp.name)),
    )

app = Starlette(
    routes=[
        Route("/config", put_config, methods=["PUT"]),
        Route("/config", get_config, methods=["GET"]),
        Route("/next", get_next_sample, methods=["GET"]),
        Route("/sample", get_sample_by_id, methods=["GET"]),
        Route("/annotate", handle_annotation, methods=["POST"]),
        Route("/annotate", handle_annotation, methods=["DELETE"]),
        Route("/stats", get_stats, methods=["GET"]),
        Route("/training_stats", get_training_stats, methods=["GET"]),
        Route("/export_db", export_db, methods=["GET"]),
    ]
)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


