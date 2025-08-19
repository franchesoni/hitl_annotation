from pathlib import Path
from flask import Flask, send_from_directory, request, jsonify
from functools import lru_cache
import timm

from src.new_backend.db import get_conn, get_config, update_config, get_next_sample_by_strategy
from src.new_backend.db_init import initialize_database_if_needed
from src.new_backend.utils import create_image_response
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
    print('config', config)
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

@app.get("/api/samples/<string:sample_id>")
def get_sample_by_id(sample_id: str):
    """
    Returns a sample by its ID.
    Path param: `sample_id` identifies the resource.
    """
    # TODO: fetch by `sample_id`
    raise NotImplementedError("This endpoint is not implemented yet.")

@app.put("/api/annotate/<path:filepath>")
def put_annotation(filepath: str):
    """
    Replaces an existing annotation in the database.
    Path param: `filepath` identifies the resource.
    Body: {"class": str}
    """
    data = request.get_json(silent=True) or {}
    class_ = data.get("class")
    # TODO: upsert annotation for `filepath` with label `class_`
    raise NotImplementedError("This endpoint is not implemented yet.")

@app.delete("/api/annotate/<path:filepath>")
def delete_annotation(filepath: str):
    """
    Deletes an existing annotation in the database.
    Path param: `filepath` identifies the resource.
    """
    # TODO: delete annotation for `filepath`
    raise NotImplementedError("This endpoint is not implemented yet.")

@app.get("/api/stats")
def get_stats():
    """Returns current statistics."""
    # TODO: compute and return stats
    raise NotImplementedError("This endpoint is not implemented yet.")

@app.put("/api/ai/run")
def run_ai():
    """Changes a flag in the DB that the AI script should check."""
    # TODO: set run flag
    raise NotImplementedError("This endpoint is not implemented yet.")

@app.put("/api/ai/stop")
def stop_ai():
    """Changes a flag in the DB that the AI script should check."""
    # TODO: clear run flag
    raise NotImplementedError("This endpoint is not implemented yet.")

@app.get("/api/export")
def export_data():
    """Gets the annotations from the database as a file."""
    # TODO: stream exported annotations file in your default format
    raise NotImplementedError("This endpoint is not implemented yet.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
