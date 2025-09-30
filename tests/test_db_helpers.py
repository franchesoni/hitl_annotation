import pytest


def test_upsert_annotation_requires_mask_path(test_db):
    db = test_db["db"]
    sample_id = test_db["sample_id"]

    with pytest.raises(ValueError):
        db.upsert_annotation(sample_id, "rock", "mask")


def test_upsert_annotation_normalizes_coordinates(test_db):
    db = test_db["db"]
    sample_id = test_db["sample_id"]

    db.upsert_annotation(
        sample_id,
        "granite",
        "bbox",
        col=0.1,
        row=0.2,
        width=0.3,
        height=0.4,
        timestamp=9,
    )

    annotations = db.get_annotations(sample_id)
    bbox = [a for a in annotations if a["type"] == "bbox"][0]

    assert bbox["col01"] == db.to_ppm(0.1)
    assert bbox["row01"] == db.to_ppm(0.2)
    assert bbox["width01"] == db.to_ppm(0.3)
    assert bbox["height01"] == db.to_ppm(0.4)
    assert bbox["timestamp"] == 9


def test_delete_annotations_by_type_isolated(test_db):
    db = test_db["db"]
    sample_id = test_db["sample_id"]

    db.upsert_annotation(sample_id, "cat", "label", timestamp=1)
    db.upsert_annotation(sample_id, "dog", "point", col=0.1, row=0.2, timestamp=1)

    deleted = db.delete_annotations_by_type(sample_id, "label")
    assert deleted == 1

    annotations = db.get_annotations(sample_id)
    assert all(a["type"] != "label" for a in annotations)
    assert any(a["type"] == "point" for a in annotations)


def test_clear_point_annotations_only_points(test_db):
    db = test_db["db"]
    sample_id = test_db["sample_id"]

    db.upsert_annotation(sample_id, "owl", "label", timestamp=2)
    db.upsert_annotation(sample_id, "sparrow", "point", col=0.3, row=0.5, timestamp=2)

    cleared = db.clear_point_annotations(sample_id)
    assert cleared == 1

    annotations = db.get_annotations(sample_id)
    assert any(a["type"] == "label" for a in annotations)
    assert all(a["type"] != "point" for a in annotations)
