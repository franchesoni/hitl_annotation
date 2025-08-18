import mimetypes
from flask import send_file
from . import db

def create_image_response(sample_info):
    """Helper function to create image response with headers
    
    Args:
        sample_info: Dict with 'id' and 'sample_filepath' keys
    """
    sample_id = sample_info["id"]
    sample_filepath = sample_info["sample_filepath"]
    
    mime_type, _ = mimetypes.guess_type(sample_filepath)
    if mime_type is None:
        mime_type = "application/octet-stream"

    # Get label or prediction data using sample_id (faster lookup)
    anns = db.get_annotations(sample_id)
    label_ann = next((a for a in anns if a.get('type') == 'label'), None)

    headers = {
        "X-Image-Id": str(sample_id),  # Always include the image ID
        "X-Image-Filepath": str(sample_filepath)  # include the image filepath
    }
    if label_ann:
        headers["X-Label-Class"] = str(label_ann.get('class', ''))
        headers["X-Label-Source"] = "annotation"
    else:
        # Use prediction if available; there should be at most one per image
        preds = db.get_predictions(sample_id)
        pred_candidates = [
            p
            for p in preds
            if p.get("type") == "label" and p.get("probability") is not None
        ]
        assert len(pred_candidates) <= 1, "Expected at most one prediction per image"
        if pred_candidates:
            pred_ann = pred_candidates[0]
            headers["X-Label-Class"] = str(pred_ann.get("class", ""))
            headers["X-Label-Source"] = "prediction"
            headers["X-Label-Probability"] = str(pred_ann.get("probability", ""))

    # Use Flask's send_file with custom headers
    response = send_file(sample_filepath, mimetype=mime_type)
    for key, value in headers.items():
        response.headers[key] = value
    
    return response


