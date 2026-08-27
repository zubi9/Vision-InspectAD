from pathlib import Path

from src.anomalib_pipeline.prepare_data import verify_category


def _touch_pngs(dir_path: Path, n: int):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (dir_path / f"{i:03d}.png").touch()


def test_verify_category_valid_structure(tmp_path):
    cat_root = tmp_path / "bottle"
    _touch_pngs(cat_root / "train" / "good", 3)
    _touch_pngs(cat_root / "test" / "good", 2)
    _touch_pngs(cat_root / "test" / "broken_large", 2)
    _touch_pngs(cat_root / "ground_truth" / "broken_large", 2)

    result = verify_category(cat_root)

    assert result["issues"] == []
    assert result["counts"]["train_good"] == 3
    assert result["counts"]["test_good"] == 2
    assert result["counts"]["test_anomalous"] == 2
    assert result["counts"]["masks"] == 2


def test_verify_category_missing_train_good(tmp_path):
    cat_root = tmp_path / "bottle"
    _touch_pngs(cat_root / "test" / "good", 2)

    result = verify_category(cat_root)

    assert any("train" in issue and "good" in issue for issue in result["issues"])
    assert result["counts"]["train_good"] == 0


def test_verify_category_anomalous_without_masks_flagged(tmp_path):
    """Anomalous test images with no corresponding ground_truth masks is a real
    data-integrity problem -- unsupervised AD evaluation needs pixel-level masks
    to compute pixel-AUROC/PRO, so this should be surfaced as an issue, not
    silently accepted."""
    cat_root = tmp_path / "bottle"
    _touch_pngs(cat_root / "train" / "good", 3)
    _touch_pngs(cat_root / "test" / "good", 2)
    _touch_pngs(cat_root / "test" / "broken_large", 2)
    # no ground_truth/ directory at all

    result = verify_category(cat_root)

    assert any("ground_truth" in issue for issue in result["issues"])


def test_verify_category_normal_only_needs_no_masks(tmp_path):
    """A category with zero anomalous test images legitimately has no masks --
    that's not an issue (this is the normal MVTec AD train split itself, and
    some categories' test/good-only slices in ad hoc setups)."""
    cat_root = tmp_path / "bottle"
    _touch_pngs(cat_root / "train" / "good", 3)
    _touch_pngs(cat_root / "test" / "good", 2)
    # no anomalous test images, no ground_truth dir

    result = verify_category(cat_root)

    assert result["issues"] == []
    assert result["counts"]["test_anomalous"] == 0
    assert result["counts"]["masks"] == 0


def test_verify_category_missing_test_dir(tmp_path):
    cat_root = tmp_path / "bottle"
    _touch_pngs(cat_root / "train" / "good", 3)

    result = verify_category(cat_root)

    assert any("test" in issue for issue in result["issues"])
