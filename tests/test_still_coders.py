"""Lab tests of the in-memory codec wrappers JPEG (B06), HEIF (B07), AVIF (B08): imstudies.lab.coders.

Need Pillow 12.3.0 and pillow-heif 1.8.1 of the [lab] group. Every step of the validation order
and every promise of the outputs is checked on both sides.

The wrappers and Pillow (the [lab] group) are imported inside the tests, not at module level:
`-m core` then deselects these tests in an environment without [lab] instead of failing at
collection, and nothing is skipped silently in an environment with [lab].
"""

import hashlib
import importlib
import io

import numpy as np
import pytest

from imstudies.core.metrics import Refusal

pytestmark = pytest.mark.lab

CODECS = ("jpeg", "heif", "avif")


def _codec(name):
    """The wrapper module imstudies.lab.coders.<name> (imports Pillow or pillow-heif)."""
    return importlib.import_module(f"imstudies.lab.coders.{name}")


def _frame(h, w):
    """Recipe of frame E0 (seed 1) on (h, w)."""
    y, x = np.mgrid[0:h, 0:w]
    return (x / 2 + y / 4 + np.random.default_rng(1).normal(0, 8, (h, w))).clip(0, 255).astype(np.uint8)


def _reason(module, *args, **kwargs):
    with pytest.raises(Refusal) as info:
        module.encode_decode(*args, **kwargs)
    return info.value.reason


@pytest.mark.parametrize("name", CODECS)
def test_refusals_and_their_order(name):
    m, good = _codec(name), _frame(16, 16)
    assert _reason(m, good[None], 50, max_pixels=256) == "not_2d"
    assert _reason(m, good.astype(np.float64)[None], None, max_pixels=None) == "not_2d"
    assert _reason(m, good.astype(np.uint16), 50, max_pixels=256) == "unsupported_dtype"
    assert _reason(m, good[:7].astype(np.float64), 50, max_pixels=256) == "unsupported_dtype"
    for small in (good[:7], good[:, :7]):
        assert _reason(m, small, None, max_pixels=None) == "image_too_small"
    assert _reason(m, good, None, max_pixels=256) == "parameter_not_assigned"
    assert _reason(m, good, 50, max_pixels=None) == "parameter_not_assigned"
    for q in (50.0, "50", True, np.bool_(True)):
        assert _reason(m, good, q, max_pixels=1) == "parameter_out_of_range"  # (5) before (6)
    assert _reason(m, good, 50, max_pixels=255) == "image_too_large"
    huge = np.broadcast_to(np.uint8(7), (100_000, 100_000))
    assert _reason(m, huge, 50, max_pixels=10**6) == "image_too_large"


@pytest.mark.parametrize("name, low, high", [("jpeg", 1, 100), ("heif", 0, 100), ("avif", 0, 100)])
def test_quality_range_both_sides(name, low, high):
    m, image = _codec(name), _frame(16, 16)
    for q in (low - 1, high + 1):
        assert _reason(m, image, q, max_pixels=256) == "parameter_out_of_range"
    for q in (low, high, np.int64(50)):
        result = m.encode_decode(image, q, max_pixels=256)
        assert result.recon.shape == image.shape and len(result.file) > 0


@pytest.mark.parametrize("name", CODECS)
@pytest.mark.parametrize("shape", [(8, 8), (21, 13), (517, 389)])
def test_shape_kept_and_three_calls_identical(name, shape):
    m, image = _codec(name), _frame(*shape)
    results = [m.encode_decode(image, 50, max_pixels=image.size) for _ in range(3)]
    assert len({hashlib.sha256(r.file).hexdigest() for r in results}) == 1
    assert all(r.recon.shape == shape and r.recon.dtype == np.uint8 for r in results)
    assert all(np.array_equal(r.recon, results[0].recon) for r in results)


def test_provenance_names_versions_and_every_argument():
    jpeg, heif, avif, image = _codec("jpeg"), _codec("heif"), _codec("avif"), _frame(16, 16)
    pj = jpeg.encode_decode(image, 75, max_pixels=256).provenance
    assert pj["versions"] == jpeg.EXPECTED_VERSIONS
    assert (pj["encode"]["quality"], pj["encode"]["optimize"], pj["encode"]["progressive"]) == (75, False, False)
    ph = heif.encode_decode(image, 75, max_pixels=256).provenance
    assert ph["versions"] == heif.EXPECTED_VERSIONS and ph["hvcC_chroma_format_idc"] == 0
    assert ph["encode"]["enc_params"] == {"preset": "slow", "tune": "ssim", "tu-intra-depth": "2"}
    assert ph["options"] == {"SAVE_NCLX_PROFILE": True, "THUMBNAILS": True, "GRID_TILE_SIZE": 0}
    pa = avif.encode_decode(image, 75, max_pixels=256).provenance
    assert pa["versions"] == avif.EXPECTED_VERSIONS and pa["av1C_monochrome"] == 1
    assert {k: pa["encode"][k] for k in ("subsampling", "speed", "codec", "max_threads")} == \
        {"subsampling": "4:0:0", "speed": 6, "codec": "aom", "max_threads": 1}


def _colour_heif(image, quality):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.stack([image] * 3, axis=-1), "RGB").save(buf, "HEIF", quality=quality)
    return buf.getvalue()


def _colour_avif(image, quality):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.stack([image] * 3, axis=-1), "RGB").save(buf, "AVIF", quality=quality, subsampling="4:2:0",
                                                                 max_threads=1)
    return buf.getvalue()


def test_heif_colour_file_is_a_failure(monkeypatch):
    heif, isobmff, image = _codec("heif"), _codec("isobmff"), _frame(16, 16)
    assert isobmff.hevc_chroma_format_idc(heif.encode_decode(image, 50, max_pixels=256).file) == 0
    monkeypatch.setattr(heif, "_encode", _colour_heif)
    with pytest.raises(RuntimeError, match="hvcC chroma_format_idc = 1, not 0") as info:
        heif.encode_decode(image, 50, max_pixels=256)
    assert not isinstance(info.value, Refusal)


def test_avif_colour_file_is_a_failure(monkeypatch):
    avif, isobmff, image = _codec("avif"), _codec("isobmff"), _frame(16, 16)
    assert isobmff.av1_monochrome(avif.encode_decode(image, 50, max_pixels=256).file) == 1
    monkeypatch.setattr(avif, "_encode", _colour_avif)
    with pytest.raises(RuntimeError, match="av1C monochrome = 0, not 1"):
        avif.encode_decode(image, 50, max_pixels=256)


@pytest.mark.parametrize("name", CODECS)
def test_version_mismatch_is_refused_in_words_before_the_call(name, monkeypatch):
    m, image = _codec(name), _frame(16, 16)
    assert m.encode_decode(image, 50, max_pixels=256).provenance["versions"] == m.EXPECTED_VERSIONS
    key = next(iter(m.EXPECTED_VERSIONS))
    monkeypatch.setattr(m, "EXPECTED_VERSIONS", dict(m.EXPECTED_VERSIONS, **{key: "0.0.0"}))
    monkeypatch.setattr(m, "_encode", lambda *a: pytest.fail("the coder must not be called"))
    with pytest.raises(RuntimeError, match="library versions differ from the declared ones"):
        m.encode_decode(image, 50, max_pixels=256)
