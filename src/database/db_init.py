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
        "/home/fmarchesoni/walden/out/img1/composite_image.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_0_0.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_0_1.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_0_2.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_0_3.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_0_4.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_0_5.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_0_6.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_0_7.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_0_8.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_0_9.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_1_0.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_1_1.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_1_2.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_1_3.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_1_4.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_1_5.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_1_6.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_1_7.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_1_8.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_1_9.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_2_0.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_2_1.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_2_2.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_2_3.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_2_4.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_2_5.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_2_6.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_2_7.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_2_8.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_2_9.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_3_0.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_3_1.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_3_2.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_3_3.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_3_4.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_3_5.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_3_6.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_3_7.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_3_8.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_3_9.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_4_0.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_4_1.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_4_2.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_4_3.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_4_4.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_4_5.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_4_6.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_4_7.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_4_8.jpeg",
        "/home/fmarchesoni/walden/out/img1/img_full_crop_z8_tile_4_9.jpeg",
    ]

    db_dict = {
        "samples": [{"filepath": img} for img in IMAGES],
        "annotations": [],
        "predictions": [],
    }
    ### EDIT END
    return db_dict
