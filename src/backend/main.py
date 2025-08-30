from pathlib import Path
from flask import Flask, request, jsonify, send_file
from functools import lru_cache
import timm

from src.backend.db import (
    get_config, update_config, get_next_sample_by_strategy,
    get_sample_by_id, upsert_annotation, delete_annotation_by_sample_id,
    get_annotation_stats, export_annotations, release_claim_by_id,
    get_most_recent_prediction, store_live_accuracy, get_annotations,
    add_point_annotation, delete_point_annotation, clear_point_annotations,
    get_sample_prev_by_id, get_sample_next_by_id, get_predictions,
)
from src.backend.db_init import initialize_database_if_needed
import mimetypes

initialize_database_if_needed()

####### INITIALIZE APP #######
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
# Fail loudly: enable development-style debugging and exception propagation
app.config.update(
    ENV="development",
    DEBUG=True,
    PROPAGATE_EXCEPTIONS=True,
    TRAP_HTTP_EXCEPTIONS=True,
)


@lru_cache(maxsize=1)
def _list_architectures():
    """Return all allowed model architectures."""
    resnets = ["resnet18", "resnet34"]
    return resnets + [m for m in sorted(timm.list_models()) if m not in resnets]


def create_image_response(sample_info):
    """Create an image response with informative headers.

    Args:
        sample_info: Dict with 'id' and 'sample_filepath'.
    """
    sample_id = sample_info["id"]
    sample_filepath = sample_info["sample_filepath"]

    mime_type, _ = mimetypes.guess_type(sample_filepath)
    if mime_type is None:
        mime_type = "application/octet-stream"

    anns = get_annotations(sample_id)
    label_ann = next((a for a in anns if a.get("type") == "label"), None)

    headers = {
        "X-Image-Id": str(sample_id),
        "X-Image-Filepath": str(sample_filepath),
    }
    if label_ann:
        headers["X-Label-Class"] = str(label_ann.get("class", ""))
        headers["X-Label-Source"] = "annotation"
    else:
        preds = get_predictions(sample_id)
        pred_candidates = [
            p for p in preds if p.get("type") == "label" and p.get("probability") is not None
        ]
        assert len(pred_candidates) <= 1, "Expected at most one prediction per image"
        if pred_candidates:
            pred_ann = pred_candidates[0]
            headers["X-Label-Class"] = str(pred_ann.get("class", ""))
            headers["X-Label-Source"] = "prediction"
            headers["X-Label-Probability"] = str(pred_ann.get("probability", ""))

    response = send_file(sample_filepath, mimetype=mime_type)
    for key, value in headers.items():
        response.headers[key] = value
    return response


@app.route("/")
def index():
    return app.send_static_file("router.html")

@app.route("/classification")
def classification():
    return app.send_static_file("classification/index.html")

@app.route("/points")
def points():
    return app.send_static_file("points/index.html")

# Static files are served by Flask from FRONTEND_DIR via static_url_path=""


@app.route("/api/health", methods=["GET"])
def health():
    return {"status": "ok"}


####### API #########
@app.put("/api/config")
def put_config():
    """Updates the configuration in the db.

    Expects a JSON body with the configuration object.
    """
    config = request.get_json(silent=False) or {}
    update_config(config)
    return jsonify({"status": "Config saved successfully"})


@app.get("/api/config")  # available architectures should go here
def get_config_endpoint():
    """Gets the config from the db."""
    config = get_config()
    # Add available architectures to the config response
    config["available_architectures"] = _list_architectures()
    return jsonify(config)


@app.get("/api/samples/next")
def get_next_sample():
    """
    Returns the next sample to be annotated.
    Query param: `strategy` determines selection strategy.
    Query param: `pick` specifies class for pick_class strategy.
    """
    strategy = request.args.get("strategy")
    pick = request.args.get("pick")  # For pick_class strategy
    sample_info = get_next_sample_by_strategy(strategy, pick)
    if sample_info:
        return create_image_response(sample_info)
    else:
        return jsonify({"error": "No more samples available for annotation"}), 404


@app.get("/api/samples/<int:sample_id>")
def get_sample_by_id_endpoint(sample_id: int):
    """
    Returns a sample by its ID.
    Path param: `sample_id` identifies the resource.
    """
    sample_info = get_sample_by_id(sample_id)
    if sample_info:
        return create_image_response(sample_info)
    else:
        return jsonify({"error": f"Sample with ID {sample_id} not found"}), 404


@app.get("/api/samples/<int:sample_id>/prev")
def get_sample_prev(sample_id: int):
    """
    Returns the previous sample by ID (sample with ID < current_id).
    Path param: `sample_id` identifies the current sample.
    """
    sample_info = get_sample_prev_by_id(sample_id)
    if sample_info:
        return create_image_response(sample_info)
    else:
        return jsonify({"error": f"No previous sample found before ID {sample_id}"}), 404


@app.get("/api/samples/<int:sample_id>/next")
def get_sample_next(sample_id: int):
    """
    Returns the next sample by ID (sample with ID > current_id).
    Path param: `sample_id` identifies the current sample.
    """
    sample_info = get_sample_next_by_id(sample_id)
    if sample_info:
        return create_image_response(sample_info)
    else:
        return jsonify({"error": f"No next sample found after ID {sample_id}"}), 404


@app.put("/api/annotate/<int:sample_id>")
def put_annotation(sample_id: int):
    """
    Replaces an existing annotation in the database.
    Path param: `sample_id` identifies the resource.
    Body: {"class": str, "type": str (optional), "row": int (optional), "col": int (optional), ...}
    """
    data = request.get_json(silent=False) or {}
    class_ = data.get("class")
    if not class_:
        return jsonify({"error": "Missing required field: class"}), 400

    # Default to label type if not specified
    annotation_type = data.get("type", "label")

    # Prepare annotation data
    annotation_data = {
        "timestamp": data.get("timestamp"),
    }

    # Add coordinates if provided
    if annotation_type in ["point", "bbox"]:
        if "row" in data:
            annotation_data["row"] = data["row"]
        if "col" in data:
            annotation_data["col"] = data["col"]

    if annotation_type == "bbox":
        if "width" in data:
            annotation_data["width"] = data["width"]
        if "height" in data:
            annotation_data["height"] = data["height"]

    upsert_annotation(sample_id, class_, annotation_type, **annotation_data)

    # Check for live accuracy if this is a label annotation
    if annotation_type == "label":
        predicted_class = get_most_recent_prediction(sample_id)
        if predicted_class:
            is_correct = (predicted_class == class_)
            store_live_accuracy(sample_id, is_correct)

    # Release the claim since annotation is complete
    release_claim_by_id(sample_id)

    return jsonify({"status": "Annotation saved successfully"})


@app.delete("/api/annotate/<int:sample_id>")
def delete_annotation(sample_id: int):
    """
    Deletes an existing annotation in the database.
    Path param: `sample_id` identifies the resource.
    """
    success = delete_annotation_by_sample_id(sample_id)
    if success:
        # Release the claim since annotation is removed
        release_claim_by_id(sample_id)
        return jsonify({"status": "Annotation deleted successfully"})
    else:
        return jsonify({"error": f"No annotation found for sample ID {sample_id}"}), 404


@app.get("/api/annotations/<int:sample_id>")
def get_annotations_endpoint(sample_id: int):
    """Get all annotations for a specific sample ID."""
    annotations = get_annotations(sample_id)
    return jsonify({"annotations": annotations})


@app.post("/api/annotations/<int:sample_id>/points")
def add_point_endpoint(sample_id: int):
    """Add a point annotation to a sample."""
    data = request.get_json(silent=False) or {}
    class_name = data.get("class")
    x = data.get("x")  # normalized coordinate [0,1]
    y = data.get("y")  # normalized coordinate [0,1]
    
    if not class_name:
        return jsonify({"error": "Missing required field: class"}), 400
    if x is None or y is None:
        return jsonify({"error": "Missing required fields: x, y"}), 400
    
    add_point_annotation(sample_id, class_name, x, y)
    return jsonify({"status": "Point annotation added successfully"})


@app.delete("/api/annotations/<int:sample_id>/points")
def delete_point_endpoint(sample_id: int):
    """Delete a point annotation near the specified coordinates."""
    data = request.get_json(silent=False) or {}
    x = data.get("x")  # normalized coordinate [0,1]
    y = data.get("y")  # normalized coordinate [0,1]
    tolerance = data.get("tolerance", 0.02)  # default 2% tolerance
    
    if x is None or y is None:
        return jsonify({"error": "Missing required fields: x, y"}), 400
    
    success = delete_point_annotation(sample_id, x, y, tolerance)
    if success:
        return jsonify({"status": "Point annotation deleted successfully"})
    else:
        return jsonify({"error": "No point annotation found near the specified coordinates"}), 404


@app.delete("/api/annotations/<int:sample_id>/points/all")
def clear_points_endpoint(sample_id: int):
    """Clear all point annotations for a sample."""
    count = clear_point_annotations(sample_id)
    return jsonify({"status": f"Cleared {count} point annotations"})


@app.get("/api/stats")
def get_stats():
    """Returns current statistics."""
    stats = get_annotation_stats()
    return jsonify(stats)


@app.get("/api/export")
def export_data():
    """Gets the annotations from the database as a file."""
    annotations = export_annotations()
    return jsonify({"annotations": annotations})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
