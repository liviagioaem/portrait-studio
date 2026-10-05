from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw

from app import process_credential_framing


def _build_case1() -> Image.Image:
    # Head near top with oversized torso and dark bottom artifact.
    w, h = 800, 1000
    img = Image.new("RGBA", (w, h), (230, 232, 238, 255))
    d = ImageDraw.Draw(img)

    # Hair/head close to top edge.
    d.ellipse((280, 20, 520, 280), fill=(210, 180, 165, 255))
    d.ellipse((285, 10, 515, 120), fill=(40, 40, 40, 255))

    # Shoulders/torso too large.
    d.polygon([(190, 420), (610, 420), (730, 980), (70, 980)], fill=(70, 90, 120, 255))

    # Simulated dark artifact region at bottom.
    d.rectangle((0, 900, 800, 1000), fill=(0, 0, 0, 255))
    return img


def _build_case2() -> Image.Image:
    # Tight face + displaced person with transparency and gray halo-like artifact.
    w, h = 700, 700
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Circular crop-like subject, off-center and too large.
    d.ellipse((170, 80, 610, 540), fill=(228, 190, 174, 255))
    d.ellipse((185, 70, 600, 210), fill=(35, 35, 35, 255))

    # Gray halo artifact around subject.
    d.arc((150, 70, 630, 560), start=180, end=360, fill=(160, 160, 160, 190), width=10)

    # Shoulders.
    d.polygon([(160, 430), (620, 430), (680, 690), (130, 690)], fill=(95, 110, 140, 255))
    return img


def _build_face_only_case() -> Image.Image:
    img = Image.new("RGBA", (640, 640), (235, 236, 240, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((210, 110, 430, 350), fill=(220, 182, 165, 255))
    d.ellipse((215, 95, 425, 180), fill=(35, 35, 35, 255))
    d.rectangle((280, 320, 360, 420), fill=(220, 182, 165, 255))
    return img


def _build_sufficient_clothing_case() -> Image.Image:
    img = Image.new("RGBA", (700, 800), (236, 237, 241, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((250, 100, 450, 320), fill=(225, 188, 170, 255))
    d.ellipse((255, 90, 445, 165), fill=(30, 30, 30, 255))
    d.polygon([(140, 360), (560, 360), (640, 790), (70, 790)], fill=(40, 58, 92, 255))
    return img


def _build_partial_clothing_case() -> Image.Image:
    img = Image.new("RGBA", (700, 800), (236, 237, 241, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((250, 100, 450, 320), fill=(225, 188, 170, 255))
    d.ellipse((255, 90, 445, 165), fill=(30, 30, 30, 255))
    # Small visible shirt fragment only (partial clothing).
    d.rectangle((300, 360, 400, 470), fill=(62, 86, 124, 255))
    return img


def test_case1_regression_framing_preserves_head_margin_and_shoulders() -> None:
    image = _build_case1()

    result = process_credential_framing(
        image,
        eye_left=(360.0, 145.0),
        eye_right=(440.0, 145.0),
    )

    assert result.square.size[0] == result.square.size[1]
    assert result.status in {"approved", "review"}
    assert result.metrics["head_margin_norm"] >= 0.06
    assert result.metrics["shoulder_visibility_norm"] >= 0.10
    assert result.metrics["chin_norm"] <= 0.95


def test_case2_regression_framing_reduces_tight_crop_and_recenters() -> None:
    image = _build_case2()

    result = process_credential_framing(
        image,
        eye_left=(330.0, 250.0),
        eye_right=(460.0, 248.0),
    )

    assert result.square.size[0] == result.square.size[1]
    assert result.metrics["eye_mid_y_norm"] <= 0.48
    assert abs(result.metrics["face_center_x_norm"] - 0.5) <= 0.16
    assert result.metrics["head_margin_norm"] >= 0.05


def test_credential_endpoint_headers_with_manual_eyes() -> None:
    from fastapi.testclient import TestClient
    from app import app

    image = _build_case1().convert("RGB")
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    payload.seek(0)

    client = TestClient(app)
    response = client.post(
        "/credential-process",
        files={"file": ("case1.png", payload.getvalue(), "image/png")},
        data={
            "shape": "round",
            "eye_left_x": "360",
            "eye_left_y": "145",
            "eye_right_x": "440",
            "eye_right_y": "145",
        },
    )

    assert response.status_code == 200
    assert response.headers.get("x-credential-status") in {"approved", "review"}
    assert response.headers.get("content-type") == "image/png"


def test_clothing_generation_allowed_for_face_only_when_type_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import Image

    called = {"value": False}

    def fake_inpaint(base_rgb: Image.Image, clothing_mask: Image.Image, clothing_type: str, seed: int) -> Image.Image:
        called["value"] = True
        out = base_rgb.copy()
        d = ImageDraw.Draw(out)
        d.polygon([(140, 430), (500, 430), (560, 630), (80, 630)], fill=(55, 70, 95))
        return out

    monkeypatch.setattr("app._run_credential_clothing_inpaint", fake_inpaint)

    result = process_credential_framing(
        _build_face_only_case(),
        eye_left=(290.0, 215.0),
        eye_right=(350.0, 214.0),
        clothing_type="male_formal",
    )

    assert called["value"] is True
    assert result.clothing_status == "generated"
    assert result.clothing_generated is True


def test_clothing_is_preserved_when_sufficient(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"value": False}

    def fake_inpaint(*args, **kwargs):
        called["value"] = True
        raise AssertionError("Nao deveria gerar roupa quando ja existe roupa suficiente")

    monkeypatch.setattr("app._run_credential_clothing_inpaint", fake_inpaint)

    result = process_credential_framing(
        _build_sufficient_clothing_case(),
        eye_left=(315.0, 205.0),
        eye_right=(385.0, 204.0),
        clothing_type="male_formal",
    )

    assert called["value"] is False
    assert result.clothing_status in {"preserved", "not_required"}
    assert result.clothing_generated is False


def test_partial_clothing_generates_only_missing_region(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import Image

    def fake_inpaint(base_rgb: Image.Image, clothing_mask: Image.Image, clothing_type: str, seed: int) -> Image.Image:
        # Paint only mask region to emulate localized generation.
        base = base_rgb.convert("RGB")
        mask = clothing_mask.convert("L")
        arr = base.copy()
        arr_np = np.array(arr)
        mask_np = np.array(mask)
        arr_np[mask_np > 0] = np.array([65, 80, 105], dtype=np.uint8)
        return Image.fromarray(arr_np, mode="RGB")

    import numpy as np

    monkeypatch.setattr("app._run_credential_clothing_inpaint", fake_inpaint)

    result = process_credential_framing(
        _build_partial_clothing_case(),
        eye_left=(315.0, 205.0),
        eye_right=(385.0, 204.0),
        clothing_type="female_formal",
    )

    assert result.clothing_status == "generated"
    assert result.clothing_generated is True
    assert 0.0 < result.metrics.get("clothing_generated_area_ratio", 0.0) <= 1.0


def test_clothing_generation_failure_returns_review(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_inpaint(*args, **kwargs):
        raise RuntimeError("falha simulada")

    monkeypatch.setattr("app._run_credential_clothing_inpaint", fake_inpaint)

    result = process_credential_framing(
        _build_face_only_case(),
        eye_left=(290.0, 215.0),
        eye_right=(350.0, 214.0),
        clothing_type="male_formal",
    )

    assert result.clothing_status == "review"
    assert result.clothing_generated is False
    assert "clothing_generation_failed" in result.reasons


def test_face_region_protection_blocks_unwanted_face_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import Image
    import numpy as np

    def fake_inpaint(base_rgb: Image.Image, clothing_mask: Image.Image, clothing_type: str, seed: int) -> Image.Image:
        # Simulate a bad generator that alters the whole image including face.
        arr = np.asarray(base_rgb.convert("RGB"), dtype=np.uint8).copy()
        arr[:, :, :] = np.array([250, 80, 80], dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    monkeypatch.setattr("app._run_credential_clothing_inpaint", fake_inpaint)

    source = _build_partial_clothing_case()
    result = process_credential_framing(
        source,
        eye_left=(315.0, 205.0),
        eye_right=(385.0, 204.0),
        clothing_type="female_formal",
    )

    assert result.clothing_status in {"generated", "preserved", "not_required"}

    # Face region should stay stable even when generator output is globally altered.
    source_result = process_credential_framing(
        source,
        eye_left=(315.0, 205.0),
        eye_right=(385.0, 204.0),
        clothing_type="disabled",
    )

    before = np.asarray(source_result.square.convert("RGB"), dtype=np.float32)
    after = np.asarray(result.square.convert("RGB"), dtype=np.float32)
    # Head ROI in the normalized credential frame.
    h, w = before.shape[:2]
    x0 = int(w * 0.28)
    x1 = int(w * 0.72)
    y0 = int(h * 0.14)
    y1 = int(h * 0.58)
    delta = np.abs(after[y0:y1, x0:x1] - before[y0:y1, x0:x1]).mean()
    assert delta < 16.0
