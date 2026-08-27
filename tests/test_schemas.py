import pytest
from pydantic import ValidationError

from src.common.schemas import PredictionResult, Region, UnrecognizedResult


def test_region_requires_bbox_and_score():
    region = Region(bbox=(1.0, 2.0, 3.0, 4.0), score=0.75)
    assert region.bbox == (1.0, 2.0, 3.0, 4.0)
    assert region.label is None
    assert region.mask is None


def test_region_missing_required_field_raises():
    with pytest.raises(ValidationError):
        Region(score=0.5)  # missing bbox


def test_prediction_result_anomalib_shape_normalizes():
    """Anomalib produces regions with no label -- schema must accept that."""
    result = PredictionResult(
        defect_detected=True,
        image_score=0.87,
        regions=[Region(bbox=(0, 0, 10, 10), score=0.87, label=None)],
        source_model="anomalib",
        router_class="MVTec_bottle",
        router_confidence=0.95,
    )
    assert result.source_model == "anomalib"
    assert result.regions[0].label is None
    assert result.heatmap_overlay_base64 is None


def test_prediction_result_yolo_shape_normalizes():
    """YOLO produces regions with a class label and optional mask polygon --
    same schema, different populated fields, but both are valid PredictionResults."""
    result = PredictionResult(
        defect_detected=True,
        image_score=0.6,
        regions=[Region(bbox=(0, 0, 10, 10), score=0.6, label="scratch", mask=[[1.0, 1.0], [2.0, 2.0]])],
        source_model="yolo26-seg",
        router_class="DAGM",
        router_confidence=0.8,
    )
    assert result.source_model == "yolo26-seg"
    assert result.regions[0].label == "scratch"
    assert result.regions[0].mask == [[1.0, 1.0], [2.0, 2.0]]


def test_prediction_result_rejects_unknown_source_model():
    with pytest.raises(ValidationError):
        PredictionResult(
            defect_detected=False,
            image_score=0.1,
            regions=[],
            source_model="not-a-real-backend",
            router_class="MVTec_bottle",
            router_confidence=0.9,
        )


def test_prediction_result_round_trips_through_json():
    result = PredictionResult(
        defect_detected=False,
        image_score=0.1,
        regions=[],
        source_model="anomalib",
        router_class="MVTec_screw",
        router_confidence=0.7,
    )
    parsed = PredictionResult.model_validate_json(result.model_dump_json())
    assert parsed == result


def test_unrecognized_result_defaults():
    result = UnrecognizedResult(router_raw_class="MVTec_bottle", router_confidence=0.4)
    assert result.recognized is False
    assert "threshold" in result.message
