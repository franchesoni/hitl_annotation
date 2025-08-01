def build_initial_db_dict() -> dict:
    """
    Placeholder function to build the initial database dict for import/initialization.
    The user should edit this function to load or construct their data.

    The returned object must be a dict with the following structure:

    {
        "samples": [
            {"filepath": str},
            ...
        ],
        "annotations": [
            {
                "sample_filepath": str,  # must match a filepath in samples
                "type": "label" | "bbox" | "point",
                "class": str,
                # For "point": "row", "col" (int, non-negative)
                # For "bbox": "row", "col", "width", "height" (int, non-negative)
                # For "label": no coordinates
                # Optional: "timestamp": int
            },
            ...
        ],
        "predictions": [
            {
                "sample_filepath": str,  # must match a filepath in samples
                "type": "label" | "bbox",
                "class": str,
                # For "label": "probability" (float in [0,1])
                # For "bbox": "row", "col", "width", "height" (int, non-negative)
            },
            ...
        ]
    }

    Returns:
        db_dict (dict): The database dictionary in the strict format.
    """
    # TODO: User should edit this function to load or build their data.
    ### EDIT START 
    IMAGES = [
        "/home/franchesoni/Downloads/composite_image.jpeg",
        "/home/franchesoni/Downloads/tile_0_0.jpeg",
        
    ]
    from pathlib import Path

    db_dict = {
        "samples": [{"filepath": str(ppath)} for ppath in Path("/home/franchesoni/Downloads/mnist_png/testing").glob('**/*.png')],
        "annotations": [],
        "predictions": [],
    }
    ### EDIT END
    return db_dict
