from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.requests import Request
from starlette.routing import Route
from starlette.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
import anyio
import mimetypes
import os
import tempfile
from pathlib import Path
import subprocess
import atexit
import signal
from functools import lru_cache
import json
import timm

# --- Database integration ---
from src.database.data import DatabaseAPI, validate_db_dict
from src.database.db_init import build_initial_db_dict

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# Initialize the database
with DatabaseAPI() as db:
    if not db.get_samples():
        db_dict = build_initial_db_dict()
        validate_db_dict(db_dict)
        db.set_samples([s["filepath"] for s in db_dict["samples"]])
    config = db.get_config() or {}

if not db.get_state("global", "trainer_status"):
    db.set_state("global", "trainer_status", json.dumps({"status": "idle"}))


class EnsureSessionDirMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if not os.path.exists(db.db_path):
            db.reconnect()
            if not db.get_samples():
                db_dict = build_initial_db_dict()
                validate_db_dict(db_dict)
                db.set_samples([s["filepath"] for s in db_dict["samples"]])
            global config, annotation_buffer
            config = db.get_config() or {}
            annotation_buffer.clear()
        response = await call_next(request)
        return response


def terminate_ai_process():
    """Ensure the AI subprocess and its children are terminated."""
    status_json = db.get_state("global", "trainer_status")
    if not status_json:
        return False
    try:
        status = json.loads(status_json)
    except Exception:
        status = {}
    pid = status.get("pid")
    pgid = status.get("pgid", pid)
    killed = False
    if pid:
        try:
            os.killpg(int(pgid), signal.SIGTERM)
            killed = True
        except Exception:
            try:
                os.kill(int(pid), signal.SIGTERM)
                killed = True
            except Exception:
                pass
    status.update({"status": "idle"})
    status.pop("pid", None)
    status.pop("pgid", None)
    db.set_state("global", "trainer_status", json.dumps(status))
    return killed


def _cleanup(*args):
    terminate_ai_process()


atexit.register(_cleanup)
for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_sig, _cleanup)
    except Exception:
        pass


def create_image_response(image_path, db):
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

    
    db.set_state("global", "last_image_id", str(image_path))

    return FileResponse(image_path, media_type=mime_type, headers=headers)


def get_sample_by_id(request):
    # Get id from query parameter, must be in the database
    id_param = request.query_params.get("id")
    with DatabaseAPI() as db:
        all_images = db.get_samples()
        if id_param not in all_images:
            # this check is important, without it the client can access any file in the system
            raise ValueError(f"Image with id '{id_param}' not found.")
        return create_image_response(id_param, db)



def get_next_sample(request):
    current_id = request.query_params.get("current_id")
    strategy = request.query_params.get("strategy", "least_confident_minority")

    with DatabaseAPI() as db:
        # Sequential strategy (existing behaviour)
        def sequential_next():
            filepath = db.get_next_unlabeled_sequential(current_id)
            if filepath:
                return create_image_response(filepath, db)
            return JSONResponse({"error": "No unlabeled images available"}, status_code=404)

        if strategy == "sequential":
            return sequential_next()
        if strategy == "least_confident_minority":
            filepath = db.get_next_unlabeled_default(current_id)
            if filepath:
                return create_image_response(filepath, db)
            return sequential_next()
        if strategy == "last_class":
            events = db.get_recent_events("global", "annotation", 20)
            if events:
                target_class = events[-1].get("class")
                filepath = db.get_next_unlabeled_for_class(target_class, current_id)
                if filepath:
                    return create_image_response(filepath, db)
            # Fallbacks if no sample found for last class
            filepath = db.get_next_unlabeled_default(current_id)
            if filepath:
                return create_image_response(filepath, db)
            return sequential_next()
        if strategy == "specific_class":
            target_class = request.query_params.get("class")
            if target_class:
                filepath = db.get_next_unlabeled_for_class(target_class, current_id)
            if filepath:
                return create_image_response(filepath, db)
            return sequential_next()
        return JSONResponse({"error": "Invalid strategy"}, status_code=400)


def handle_annotation(request: Request):
    with DatabaseAPI() as db:
        if request.method == "POST":
            data = anyio.from_thread.run(request.json)
            filepath = data.get("filepath")
            class_name = data.get("class")
            if not filepath or not isinstance(class_name, str):
                return JSONResponse({"error": "Missing or invalid 'filepath' or 'class'"}, status_code=400)
            # Ensure the filepath exists in the database and on disk
            if filepath not in db.get_samples() or not os.path.isfile(filepath):
                return JSONResponse({"error": "Filepath not found"}, status_code=404)
            # Write annotation to DB
            db.save_label_annotation(filepath, class_name)

            # Log annotation event
            db.log_event("global", "annotation", {"filepath": filepath, "class": class_name})

            # Accuracy tracking: check prediction
            preds = db.get_predictions(filepath)
            pred_ann = next((p for p in preds if p.get('type') == 'label' and p.get('probability') is not None), None)
            if pred_ann:
                was_correct = str(pred_ann.get("class")) == str(class_name)
                db.log_accuracy(was_correct)
            return JSONResponse({"status": "ok"})
        elif request.method == "DELETE":
            data = anyio.from_thread.run(request.json)
            filepath = data.get("filepath")
            if not filepath:
                return JSONResponse({"error": "Missing 'filepath'"}, status_code=400)
            if filepath not in db.get_samples() or not os.path.isfile(filepath):
                return JSONResponse({"error": "Filepath not found"}, status_code=404)
            db.delete_label_annotation(filepath)
            db.delete_event_by_field("global", "annotation", "filepath", filepath)
            return JSONResponse({"status": "deleted"})
        else:
            return JSONResponse({"error": "Method not allowed"}, status_code=405)

def get_accuracy_stats(request: Request):
    with DatabaseAPI() as db:
        stats = db.get_accuracy_counts()
        tries = stats["tries"]
        correct = stats["correct"]
        accuracy = (correct / tries) if tries > 0 else None
        return JSONResponse({
            "tries": tries,
            "correct": correct,
            "accuracy": accuracy
        })

def put_config(request: Request):
    data = anyio.from_thread.run(request.json)
    if not isinstance(data, dict):
        return JSONResponse({"error": "Invalid config format"}, status_code=400)

    arch = data.get("architecture")
    if arch and arch not in _list_architectures():
        return JSONResponse(
            {"error": f"Unsupported architecture '{arch}'"}, status_code=400
        )

    # Merge and save the config in the database
    config.update(data)
    with DatabaseAPI() as db:
        db.update_config(config)
    return JSONResponse({"status": "Config saved successfully"})

def get_config(request: Request):
    with DatabaseAPI() as db:
        cfg = db.get_config()
    # Treat an empty configuration as a valid result instead of returning 404.
    # The database returns an empty dict when no config has been saved yet, so
    # simply return that to the client.
    return JSONResponse(cfg)

def get_stats(request: Request):
    annotated = db.count_labeled_samples()
    total = db.count_total_samples()
    ann_counts = db.get_annotation_counts()
    pct_param = request.query_params.get("pct")
    pct = float(pct_param) if pct_param is not None else 100.0
    with DatabaseAPI() as db:
        annotated = db.count_labeled_samples()
        total = db.count_total_samples()
        ann_counts = db.get_annotation_counts()
        stats = db.get_accuracy_counts() if pct >= 100 else db.get_accuracy_recent_pct(pct)
    tries = stats["tries"]
    correct = stats["correct"]
    accuracy = (correct / tries) if tries > 0 else None
    error = (1 - accuracy) if accuracy is not None else None
    return JSONResponse(
        {
            "image": db.get_state("global", "last_image_id"),
            "annotated": annotated,
            "total": total,
            "tries": tries,
            "correct": correct,
            "accuracy": accuracy,
            "error": error,
            "annotation_counts": ann_counts,
        }
    )

def get_training_stats(request: Request):
    with DatabaseAPI() as db:
        stats = db.get_training_stats()
    return JSONResponse(stats)


@lru_cache(maxsize=1)
def _list_architectures():
    """Return all allowed model architectures."""
    resnets = ["resnet18", "resnet34"]
    return resnets + [m for m in sorted(timm.list_models()) if m not in resnets]


def get_architectures(request: Request):
    """Return available model architectures for the training process."""
    return JSONResponse({"architectures": _list_architectures()})


def run_ai(request: Request):
    """Launch the background training process if not already running."""
    status_json = db.get_state("global", "trainer_status")
    if status_json:
        try:
            status = json.loads(status_json)
            if status.get("status") == "running" and status.get("pid"):
                try:
                    os.kill(int(status.get("pid")), 0)
                    return JSONResponse({"status": "already running"}, status_code=400)
                except Exception:
                    pass
        except Exception:
            pass

    data = anyio.from_thread.run(request.json)
    arch = data.get("architecture", config.get("architecture", "resnet18"))
    if arch not in _list_architectures():
        return JSONResponse(
            {"error": f"Unsupported architecture '{arch}'"}, status_code=400
        )
    sleep = int(data.get("sleep", 0))
    budget = int(data.get("budget", 1000))
    preproc = config.get("preprocessing", {})
    resize = int(preproc.get("resize", 64))
    flip = preproc.get("flip", True)
    max_rotate = float(preproc.get("max_rotate", 10.0))
    cmd = [
        "python",
        "-m",
        "src.ml.fastai_training",
        "--arch",
        str(arch),
        "--sleep",
        str(sleep),
        "--budget",
        str(budget),
        "--resize",
        str(resize),
    ]
    if not flip:
        cmd.append("--no-flip")
    if max_rotate != 10.0:
        cmd.extend(["--max-rotate", str(max_rotate)])

    def _set_death_sig():
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.prctl(1, signal.SIGTERM)
        except Exception:
            pass

    ai_process = subprocess.Popen(
        cmd,
        start_new_session=True,
        preexec_fn=_set_death_sig,
    )
    status = {
        "pid": ai_process.pid,
        "pgid": os.getpgid(ai_process.pid),
        "status": "running",
    }
    db.set_state("global", "trainer_status", json.dumps(status))
    return JSONResponse({"status": "started"})


def stop_ai(request: Request):
    """Terminate the background training process if running."""
    if terminate_ai_process():
        return JSONResponse({"status": "stopped"})
    return JSONResponse({"status": "not running"}, status_code=400)


def export_db(request: Request):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.close()
    with DatabaseAPI() as db:
        db.export_db_as_json(tmp.name)
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
        Route("/architectures", get_architectures, methods=["GET"]),
        Route("/run_ai", run_ai, methods=["POST"]),
        Route("/stop_ai", stop_ai, methods=["POST"]),
        Route("/export_db", export_db, methods=["GET"]),
    ],
    middleware=[Middleware(EnsureSessionDirMiddleware)],
)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


