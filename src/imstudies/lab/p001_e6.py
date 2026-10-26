"""E6 of P001: the pages for the owner's eyes - the original fragment and the four codecs at their threshold PCC of
PSNR-HVS-M 42 dB side by side, at 1 : 1 (STUDY.md section 6, «Объявление E6»; ../REFERENCE.md#lab-pages).

Three things live here, so that the run, its tests and the bench share them: the declared choice of 14 fragments
from the features of the E5 run file; the one-sentence claim of a page and its check against the run-file entry,
field by field; the page itself - one function for the page and the bench. The page is an 8-bit grayscale PNG:
the panels are pasted pixel for pixel (no interpolation, no smoothing, no level stretch); text is DejaVu Sans of
matplotlib, so the page looks the same on every machine.
"""

import re
from pathlib import Path

import numpy as np

from imstudies.lab.runs import RunRefusal

CODECS = ("vtm", "jpeg", "heif", "avif")  # the order of the panels after the original
NAMES = {"vtm": "VVC (VTM)", "jpeg": "JPEG", "heif": "HEIF", "avif": "AVIF"}
THRESHOLD_DB = 42.0
LISTS = (  # (number, feature key of analysis.complexity.fragments, words, side, count) - STUDY.md section 6
    (1, "mgm", "наибольший MGM", "max", 3),
    (2, "mgm", "наименьший MGM", "min", 3),
    (3, "ln_cr_vtm", "наибольший ln CR VVC в пороге", "max", 3),
    (4, "ln_r_jpeg", "наибольший ln r VVC / JPEG", "max", 3),
)
SEED, N_RANDOM = 20261024, 2
RANDOM_WORDS = "случайный выбор из оставшихся (numpy, seed 20261024)"
CONCLUSION = "у всех четырёх кодеков PSNR-HVS-M не ниже 42 дБ — по метрике искажения незаметны"
EXPLANATION = (
    "Сравнивайте каждую из четырёх сжатых панелей с исходной слева в одном и том же месте: края зубов, корни, кость, "
    "пломбы.",
    "Ок — при масштабе 1 : 1 ни одна сжатая панель не отличается от исходной.",
    "Не судятся подписи панелей, разница CR между кодеками и рамки панелей.",
    "Если отличие видно хотя бы на одной сжатой панели — пишите «пропуск», а в «Примечании» — кодек и место.",
)


def select(fragments):
    """The 14 fragments of the declaration: lists 1-4 in order, a fragment chosen before is skipped (the next of the
    same list is taken), equal values - the smaller code first; then 2 at random from the remaining codes."""
    chosen, taken = [], set()
    for number, key, words, side, count in LISTS:
        known = [f for f in fragments if f.get(key) is not None]
        order = sorted(known, key=lambda f: ((-f[key] if side == "max" else f[key]), f["unit"]))
        added = 0
        for f in order:
            if added == count:
                break
            if f["unit"] in taken:
                continue
            chosen.append({"unit": f["unit"], "image": f["image"], "list": number, "feature": words, "feature_key": key,
                           "value": f[key]})
            taken.add(f["unit"])
            added += 1
    rest = sorted(f["unit"] for f in fragments if f["unit"] not in taken)
    images = {f["unit"]: f["image"] for f in fragments}
    for unit in np.random.default_rng(SEED).choice(rest, N_RANDOM, replace=False):
        chosen.append({"unit": str(unit), "image": images[str(unit)], "list": 5, "feature": RANDOM_WORDS,
                       "feature_key": None, "value": None})
    return chosen


def number(x, digits):
    return f"{x:.{digits}f}".replace(".", ",")


def quality_text(c):
    return "без потерь" if c["status"] == "lossless" else f"{number(c['psnr_hvs_m_db'], 2)} дБ"


def feature_text(page):
    return page["feature"] if page["value"] is None else f"{page['feature']} = {number(page['value'], 4)}"


def below(page):
    return [NAMES[c["codec"]] for c in page["codecs"] if c["status"] != "lossless" and c["psnr_hvs_m_db"] < THRESHOLD_DB]


def statement(page):
    """The claim of a page - one sentence from the run-file entry only (LAB-PAGES 3, the declared form)."""
    parts = [f"{NAMES[c['codec']]}: PCC {c['pcc']}, PSNR-HVS-M {quality_text(c)}, CR {number(c['cr'], 2)}"
             for c in page["codecs"]]
    low = below(page)
    conclusion = CONCLUSION if not low else f"у {', '.join(low)} PSNR-HVS-M ниже 42 дБ — по метрике искажения заметны"
    return (f"Фрагмент {page['unit']} (снимок {page['image']}), список {page['list']} — {feature_text(page)}; "
            f"{'; '.join(parts)}; {conclusion}.")


CLAIM_RE = re.compile(r"^Фрагмент (\S+) \(снимок (\S+)\), список (\d) — (.+?); (VVC \(VTM\): .+?); (JPEG: .+?); "
                      r"(HEIF: .+?); (AVIF: .+?); (у .+)\.$")
PART_RE = re.compile(r"^(VVC \(VTM\)|JPEG|HEIF|AVIF): PCC (-?\d+), PSNR-HVS-M (без потерь|\d+,\d\d дБ), CR (\d+,\d\d)$")


def check_statement(text, page):
    """The claim against the run-file entry, field by field (LAB-PAGES 3, G21): a field in the wrong place - a refusal
    naming the field."""
    m = CLAIM_RE.match(text)
    if not m:
        raise RunRefusal(f"{page['unit']}: the page claim does not have the declared form")
    want = [("fragment", page["unit"]), ("image", page["image"]), ("list", str(page["list"])),
            ("feature", feature_text(page))]
    got = [("fragment", m[1]), ("image", m[2]), ("list", m[3]), ("feature", m[4])]
    for k, c in enumerate(page["codecs"]):
        part = PART_RE.match(m[5 + k])
        if not part:
            raise RunRefusal(f"{page['unit']}: the part {k + 1} of the claim does not have the declared form")
        got += [(f"{c['codec']} name", part[1]), (f"{c['codec']} PCC", part[2]), (f"{c['codec']} PSNR-HVS-M", part[3]),
                (f"{c['codec']} CR", part[4])]
        want += [(f"{c['codec']} name", NAMES[c["codec"]]), (f"{c['codec']} PCC", str(c["pcc"])),
                 (f"{c['codec']} PSNR-HVS-M", quality_text(c)), (f"{c['codec']} CR", number(c["cr"], 2))]
    low = below(page)
    got.append(("conclusion", m[9]))
    want.append(("conclusion", CONCLUSION if not low else
                 f"у {', '.join(low)} PSNR-HVS-M ниже 42 дБ — по метрике искажения заметны"))
    wrong = [g[0] for g, w in zip(got, want) if g != w]
    if wrong:
        raise RunRefusal(f"{page['unit']}: the page claim differs from the run file in {wrong}")


def title(page):
    return (f"Фрагмент {page['unit']} (снимок {page['image']}, ортопантомограмма): исходный и четыре кодека в пороге "
            f"PSNR-HVS-M 42 дБ, масштаб 1 : 1; выбран по списку {page['list']} — {feature_text(page)}; числа — файл E5 "
            "цикла 15, в статью ещё не вошли")


def panel_labels(page):
    """Two lines over each panel: the original, then each codec with PCC, PSNR-HVS-M and CR."""
    labels = [("Исходный", f"фрагмент {page['unit']}")]
    for c in page["codecs"]:
        labels.append((f"{NAMES[c['codec']]}, PCC {c['pcc']}", f"PSNR-HVS-M {quality_text(c)}, CR {number(c['cr'], 2)}"))
    return labels


MARGIN, GAP, LINE = 16, 16, 6  # pixels


def _font(size):
    from matplotlib import font_manager
    from PIL import ImageFont

    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"), fallback_to_default=False)
    return ImageFont.truetype(path, size)


def _wrap(draw, text, font, width):
    lines, line = [], ""
    for word in text.split(" "):
        candidate = word if not line else f"{line} {word}"
        if line and draw.textlength(candidate, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + [line]


def draw_page(original, panels, labels, page_title, claim, path=None):
    """The page and the bench view of one fragment: `original` and `panels` (uint8 arrays of one shape) pasted pixel
    for pixel in a row, two label lines over each, the title above, the claim and the four phrases «На пальцах»
    below. Returns (the 8-bit grayscale image, the texts put on it, the top-left corner of every panel); saves PNG
    to `path` if given."""
    from PIL import Image, ImageDraw

    arrays = [original, *panels]
    h, w = original.shape
    for a in arrays:
        if a.dtype != np.uint8 or a.shape != (h, w):
            raise RunRefusal(f"a panel is {a.dtype} {a.shape}, the page needs uint8 {(h, w)}")
    big, small = _font(22), _font(17)
    width = MARGIN * 2 + len(arrays) * w + (len(arrays) - 1) * GAP
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    head = _wrap(probe, page_title, big, width - 2 * MARGIN)
    texts = [page_title, claim, *(f"На пальцах: {EXPLANATION[0]}" if k == 0 else EXPLANATION[k] for k in range(4))]
    foot = [line for text in texts[1:] for line in _wrap(probe, text, small, width - 2 * MARGIN)]
    head_h, label_h, foot_h = len(head) * (22 + LINE), 2 * (17 + LINE), len(foot) * (17 + LINE)
    top = MARGIN + head_h + MARGIN + label_h
    height = top + h + MARGIN + foot_h + MARGIN
    canvas = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(canvas)
    y = MARGIN
    for line in head:
        draw.text((MARGIN, y), line, fill=0, font=big)
        y += 22 + LINE
    corners = []
    for k, (a, (first, second)) in enumerate(zip(arrays, labels)):
        x = MARGIN + k * (w + GAP)
        draw.text((x, top - label_h), first, fill=0, font=small)
        draw.text((x, top - label_h + 17 + LINE), second, fill=0, font=small)
        canvas.paste(Image.fromarray(np.ascontiguousarray(a), "L"), (x, top))
        corners.append((x, top))
        texts += [first, second]
    y = top + h + MARGIN
    for line in foot:
        draw.text((MARGIN, y), line, fill=0, font=small)
        y += 17 + LINE
    if path is not None:
        canvas.save(Path(path), "PNG")
    return canvas, texts, corners
