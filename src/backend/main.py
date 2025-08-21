from pathlib import Path
from flask import Flask, send_from_directory, request, jsonify
from functools import lru_cache
import timm

from src.backend.db import (
    get_config, update_config, get_next_sample_by_strategy,
    get_sample_by_id, upsert_annotation, delete_annotation_by_sample_id,
    get_annotation_stats, export_annotations, release_claim_by_id,
    get_most_recent_prediction, store_live_accuracy,
)
from src.backend.db_init import initialize_database_if_needed
from src.backend.utils import create_image_response

initialize_database_if_needed()

####### INITIALIZE APP #######
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.config.update(DEBUG=False)


@lru_cache(maxsize=1)
def _list_architectures():
    """Return all allowed model architectures."""
    resnets = ["resnet18", "resnet34"]
    return resnets + [m for m in sorted(timm.list_models()) if m not in resnets]


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return {"status": "ok"}


####### API #########
@app.put("/api/config")
def put_config():
    """Updates the configuration in the db.

    Expects a JSON body with the configuration object.
    """
    config = request.get_json(silent=True) or {}
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


@app.put("/api/annotate/<int:sample_id>")
def put_annotation(sample_id: int):
    """
    Replaces an existing annotation in the database.
    Path param: `sample_id` identifies the resource.
    Body: {"class": str, "type": str (optional), "row": int (optional), "col": int (optional), ...}
    """
    data = request.get_json(silent=True) or {}
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
    app.run(host="0.0.0.0", port=8000)
