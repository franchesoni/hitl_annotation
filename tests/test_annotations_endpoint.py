import pytest


def _get_claimed(db_module, sample_id):
    with db_module._get_conn() as conn:  # pylint: disable=protected-access
        return conn.execute(
            "SELECT claimed FROM samples WHERE id = ?", (sample_id,)
        ).fetchone()[0]


def test_put_annotations_rejects_unsupported_type(flask_client):
    client, _, env = flask_client
    sample_id = env["sample_id"]

    response = client.put(
        f"/api/annotations/{sample_id}",
        json=[{"type": "polygon", "class": "dog"}],
    )

    assert response.status_code == 400
    assert "Unsupported annotation type" in response.get_json()["error"]
    assert env["db"].get_annotations(sample_id) == []


def test_put_annotations_accepts_skip_without_class(flask_client):
    client, _, env = flask_client
    sample_id = env["sample_id"]

    with env["db"]._get_conn() as conn:  # pylint: disable=protected-access
        conn.execute("UPDATE samples SET claimed = 1 WHERE id = ?", (sample_id,))

    response = client.put(
        f"/api/annotations/{sample_id}",
        json=[{"type": "skip", "timestamp": 123}],
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {"ok": True, "count": 1}

    annotations = env["db"].get_annotations(sample_id)
    assert len(annotations) == 1
    ann = annotations[0]
    assert ann["type"] == "skip"
    assert ann.get("class") is None
    assert ann.get("timestamp") == 123

    assert _get_claimed(env["db"], sample_id) == 0


def test_put_annotations_overwrites_by_type(flask_client):
    client, _, env = flask_client
    sample_id = env["sample_id"]

    env["db"].upsert_annotation(sample_id, "cat", "label", timestamp=1)
    env["db"].upsert_annotation(sample_id, "cat", "point", col=0.1, row=0.2, timestamp=1)

    response = client.put(
        f"/api/annotations/{sample_id}",
        json=[
            {"type": "label", "class": "dog", "timestamp": 5},
            {
                "type": "point",
                "class": "dog",
                "col": 0.3,
                "row": 0.4,
                "timestamp": 6,
            },
        ],
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {"ok": True, "count": 2}

    annotations = env["db"].get_annotations(sample_id)
    labels = [a for a in annotations if a["type"] == "label"]
    points = [a for a in annotations if a["type"] == "point"]

    assert len(labels) == 1
    assert labels[0]["class"] == "dog"
    assert labels[0]["timestamp"] == 5

    assert len(points) == 1
    assert points[0]["class"] == "dog"
    assert points[0]["col01"] == env["db"].to_ppm(0.3)
    assert points[0]["row01"] == env["db"].to_ppm(0.4)
    assert points[0]["timestamp"] == 6


def test_put_annotations_records_live_accuracy(flask_client):
    client, _, env = flask_client
    sample_id = env["sample_id"]

    with env["db"]._get_conn() as conn:  # pylint: disable=protected-access
        conn.execute(
            """
            INSERT INTO predictions (sample_id, class, type, probability, timestamp)
            VALUES (?, ?, 'label', ?, ?)
            """,
            (sample_id, "husky", env["db"].to_ppm(0.8), 10),
        )

    response = client.put(
        f"/api/annotations/{sample_id}",
        json=[{"type": "label", "class": "husky", "timestamp": 15}],
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {"ok": True, "count": 1}

    with env["db"]._get_conn() as conn:  # pylint: disable=protected-access
        rows = conn.execute(
            "SELECT value FROM curves WHERE curve_name = 'live_accuracy'"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(1.0)
