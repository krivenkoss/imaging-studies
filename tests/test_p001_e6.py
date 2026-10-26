"""Lab tests of imstudies.lab.p001_e6 (E6 of P001, STUDY.md section 6) on synthetic features and arrays.

- the choice: features made by hand so that the lists overlap and values tie -> the known 14 codes in order; a
  fragment chosen before is skipped, the next of the same list is taken; equal values - the smaller code first;
  the last two - the declared numpy call on the remaining codes;
- the page: panels at 1 : 1 - the pixels of every panel equal its input exactly; 8-bit grayscale, also after PNG;
  the title, the claim and the four phrases «На пальцах» are on it;
- the claim against its entry, field by field: built and accepted; CR or PCC of two codecs swapped, a wrong
  conclusion - refused, naming the fields.
"""

import numpy as np
import pytest

from imstudies.lab import p001_e6
from imstudies.lab.runs import RunRefusal

pytestmark = pytest.mark.lab


def unit(i):
    return f"D{i:02d}_r00_c00"


def features():
    """20 fragments: list 1 - U1, U2 tie at 0.9 (U1 first), U3; list 2 - U20, then U18 and U19 tied; list 3 - U1 is
    the largest but taken, then U4, U5, U6; list 4 - U2 and U20 the largest but taken, then U7, U8, U9."""
    out = []
    for i in range(1, 21):
        out.append({"unit": unit(i), "image": f"D{i:02d}", "mgm": 0.5 - 0.01 * i,
                    "ln_cr_vtm": 1.0 - 0.01 * i, "ln_r_jpeg": 0.1 - 0.001 * i})
    by = {f["unit"]: f for f in out}
    by[unit(1)]["mgm"] = by[unit(2)]["mgm"] = 0.9
    by[unit(3)]["mgm"] = 0.8
    by[unit(20)]["mgm"], by[unit(18)]["mgm"], by[unit(19)]["mgm"] = 0.05, 0.1, 0.1
    for i, v in ((1, 5.0), (4, 4.0), (5, 3.9), (6, 3.8)):
        by[unit(i)]["ln_cr_vtm"] = v
    for i, v in ((2, 2.0), (20, 1.9), (7, 1.0), (8, 0.9), (9, 0.8)):
        by[unit(i)]["ln_r_jpeg"] = v
    return out


def test_choice_by_the_declared_lists():
    picks = p001_e6.select(features())
    expected = [(1, 1), (2, 1), (3, 1), (20, 2), (18, 2), (19, 2), (4, 3), (5, 3), (6, 3), (7, 4), (8, 4), (9, 4)]
    assert [(p["unit"], p["list"]) for p in picks[:12]] == [(unit(i), k) for i, k in expected]
    rest = sorted(unit(i) for i in range(10, 18))
    drawn = [str(u) for u in np.random.default_rng(20261024).choice(rest, 2, replace=False)]
    assert [p["unit"] for p in picks[12:]] == drawn and [p["list"] for p in picks[12:]] == [5, 5]
    assert len({p["unit"] for p in picks}) == 14
    assert picks[3]["feature"] == "наименьший MGM" and picks[3]["value"] == 0.05 and picks[12]["value"] is None


def page_entry():
    codecs = [{"codec": c, "pcc": pcc, "status": "valid", "psnr_hvs_m_db": hvs, "cr": cr}
              for c, pcc, hvs, cr in (("vtm", 27, 42.123, 12.345), ("jpeg", 63, 42.01, 8.1), ("heif", 55, 43.5, 10.99),
                                      ("avif", 60, 42.4, 11.5))]
    return {"unit": "D05_r00_c01", "image": "D05", "list": 1, "feature": "наибольший MGM", "value": 0.43219,
            "codecs": codecs}


def test_claim_built_and_checked_field_by_field():
    page = page_entry()
    claim = p001_e6.statement(page)
    assert claim == ("Фрагмент D05_r00_c01 (снимок D05), список 1 — наибольший MGM = 0,4322; VVC (VTM): PCC 27, "
                     "PSNR-HVS-M 42,12 дБ, CR 12,35; JPEG: PCC 63, PSNR-HVS-M 42,01 дБ, CR 8,10; HEIF: PCC 55, "
                     "PSNR-HVS-M 43,50 дБ, CR 10,99; AVIF: PCC 60, PSNR-HVS-M 42,40 дБ, CR 11,50; у всех четырёх "
                     "кодеков PSNR-HVS-M не ниже 42 дБ — по метрике искажения незаметны.")
    p001_e6.check_statement(claim, page)
    swapped = page_entry()
    swapped["codecs"][0]["cr"], swapped["codecs"][1]["cr"] = page["codecs"][1]["cr"], page["codecs"][0]["cr"]
    with pytest.raises(RunRefusal, match=r"differs from the run file in \['vtm CR', 'jpeg CR'\]"):
        p001_e6.check_statement(p001_e6.statement(swapped), page)
    swapped = page_entry()
    swapped["codecs"][2]["pcc"], swapped["codecs"][3]["pcc"] = 60, 55
    with pytest.raises(RunRefusal, match=r"\['heif PCC', 'avif PCC'\]"):
        p001_e6.check_statement(p001_e6.statement(swapped), page)
    with pytest.raises(RunRefusal, match=r"\['conclusion'\]"):
        p001_e6.check_statement(claim.replace("не ниже 42 дБ", "не ниже 41 дБ"), page)


def test_claim_names_a_codec_below_and_lossless():
    page = page_entry()
    page["codecs"][1]["psnr_hvs_m_db"] = 41.5
    page["codecs"][3] |= {"status": "lossless", "psnr_hvs_m_db": "+inf"}
    claim = p001_e6.statement(page)
    assert "AVIF: PCC 60, PSNR-HVS-M без потерь, CR 11,50" in claim
    assert claim.endswith("у JPEG PSNR-HVS-M ниже 42 дБ — по метрике искажения заметны.")
    p001_e6.check_statement(claim, page)


def test_page_panels_are_the_inputs_pixel_for_pixel(tmp_path):
    rng = np.random.default_rng(1)
    original = rng.integers(0, 256, (16, 24)).astype(np.uint8)
    panels = [rng.integers(0, 256, (16, 24)).astype(np.uint8) for _ in range(4)]
    page = page_entry()
    labels = p001_e6.panel_labels(page)
    canvas, texts, corners = p001_e6.draw_page(original, panels, labels, p001_e6.title(page), "утверждение",
                                               tmp_path / "page.png")
    assert canvas.mode == "L" and len(corners) == 5
    pixels = np.asarray(canvas)
    for (x, y), a in zip(corners, [original, *panels]):
        assert np.array_equal(pixels[y:y + 16, x:x + 24], a)
    from PIL import Image
    with Image.open(tmp_path / "page.png") as png:
        assert png.mode == "L" and np.array_equal(np.asarray(png), pixels)
    assert texts[0] == p001_e6.title(page) and texts[1] == "утверждение"
    assert texts[2] == f"На пальцах: {p001_e6.EXPLANATION[0]}" and tuple(texts[3:6]) == p001_e6.EXPLANATION[1:]
    assert ("VVC (VTM), PCC 27", "PSNR-HVS-M 42,12 дБ, CR 12,35") == tuple(texts[8:10])
    with pytest.raises(RunRefusal, match="the page needs uint8"):
        p001_e6.draw_page(original, [panels[0].astype(np.uint16)], labels, "t", "c")
