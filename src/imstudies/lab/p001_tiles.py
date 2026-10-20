"""Grid of 512 x 512 tiles of the P001 copy and the share of background and frame in every tile (B11).

Contract: ../REFERENCE.md#b11 (one level above this repository). Reads data/registry/P001.json and
data/P001/DNN.png of B10 0.2.0, checks png_sha256 of every copy, lays the grid of full tiles from the
top-left corner without overlap (remainders unused) and measures per tile:

- share_white - pixels >= 250;
- share_black - pixels <= 5;
- share_flat - 8 x 8 blocks of the tile's own grid whose standard deviation is < 1.0 level
  (numpy std over the 64 values in float64, ddof = 0);
- share_any - pixels in at least one of the three masks, a pixel of a flat block counted whole.

Shares are integer counts divided by the number of pixels (blocks for share_flat). The thresholds are
measurement parameters, not a selection rule. The page for the owner's eyes - one reduced picture per
image with the grid and share_any in every tile - is drawn here, so a page and the bench use one function.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from imstudies.lab.p001 import MAX_PIXELS
from imstudies.lab.runs import RunRefusal

CONTRACT = "B11 0.1.0"
TILE, BLOCK = 512, 8
WHITE, BLACK, FLAT_STD = 250, 5, 1.0
SELECTION = (0.10, 0.25, 0.50)  # share_any thresholds counted in the book; not an exclusion rule
PAGE_SCALE = 4  # the page shows the image reduced by block means of 4 x 4 (anti-aliased)
EXPLANATION = (
    "Смотрите на числа в клетках сетки: это доля фона и рамки во фрагменте, в процентах.",
    "Ок — большие числа стоят на рамке и пустом фоне, малые — на зубах, кости и мягких тканях.",
    "Полосы у правого и нижнего края без сетки не судятся: они не входят ни в один фрагмент; правило исключения ещё не выбрано.",
    "Если на фоне или рамке стоит малое число — пишите «пропуск»; если на зубах или кости большое — «ложная тревога».",
)


def load_copy(data_root):
    """The registry and the copies of B10: every PNG present, its sha256 equal to the registry, mode L, (H, W)."""
    data_root = Path(data_root)
    registry_path = data_root / "registry" / "P001.json"
    if not registry_path.is_file():
        raise RunRefusal("the registry data/registry/P001.json of B10 0.2.0 does not exist: B11 needs the copy first")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    images = []
    for entry in registry["images"]:
        code, h, w = entry["code"], entry["H"], entry["W"]
        if h * w > MAX_PIXELS:
            raise RunRefusal(f"{code}: {h}x{w} = {h * w} pixels > the limit {MAX_PIXELS}")
        path = data_root / "P001" / f"{code}.png"
        if not path.is_file():
            raise RunRefusal(f"{code}: the copy data/P001/{code}.png does not exist")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["png_sha256"]:
            raise RunRefusal(f"{code}: png_sha256 of the copy differs from the registry")
        with Image.open(path) as im:
            mode, values = im.mode, np.asarray(im)
        if mode != "L" or values.shape != (h, w):
            raise RunRefusal(f"{code}: the copy is mode {mode} {values.shape}, the registry says L {(h, w)}")
        images.append({"code": code, "entry": entry, "values": values})
    return registry, images


def grid(h, w):
    """Full tiles from the top-left corner: (row, column, x, y) in pixels; remainders at the edges unused."""
    return [(r, c, c * TILE, r * TILE) for r in range(h // TILE) for c in range(w // TILE)]


def tile_shares(tile):
    """The four shares of one TILE x TILE uint8 tile (declared before the numbers, B11)."""
    if tile.shape != (TILE, TILE) or tile.dtype != np.uint8:
        raise ValueError(f"a tile must be uint8 {TILE}x{TILE}, got {tile.dtype} {tile.shape}")
    n_side = TILE // BLOCK
    white, black = tile >= WHITE, tile <= BLACK
    blocks = tile.reshape(n_side, BLOCK, n_side, BLOCK).swapaxes(1, 2).astype(np.float64)
    flat_blocks = blocks.std(axis=(2, 3)) < FLAT_STD
    flat = np.repeat(np.repeat(flat_blocks, BLOCK, axis=0), BLOCK, axis=1)
    n = tile.size
    return {"share_white": int(white.sum()) / n, "share_black": int(black.sum()) / n,
            "share_flat": int(flat_blocks.sum()) / flat_blocks.size, "share_any": int((white | black | flat).sum()) / n}


def measure(images):
    """One row per tile: id DNN_rRR_cCC, code, row, column, corner in pixels, four shares."""
    rows = []
    for img in images:
        for r, c, x, y in grid(*img["values"].shape):
            rows.append({"tile": f"{img['code']}_r{r:02d}_c{c:02d}", "code": img["code"], "r": r, "c": c, "x": x, "y": y,
                         **tile_shares(np.ascontiguousarray(img["values"][y:y + TILE, x:x + TILE]))})
    return rows


def per_image(registry, rows):
    """Per code: tiles in all and with share_any <= each selection share; the count must equal B10's full_tiles."""
    out = []
    for entry in registry["images"]:
        mine = [r for r in rows if r["code"] == entry["code"]]
        if len(mine) != entry["full_tiles"]:
            raise RunRefusal(f"{entry['code']}: {len(mine)} tiles, the registry says {entry['full_tiles']}")
        out.append({"code": entry["code"], "H": entry["H"], "W": entry["W"], "n_tiles": len(mine),
                    **{f"n_any_le_{round(t * 100)}": sum(r["share_any"] <= t for r in mine) for t in SELECTION},
                    "max_share_any": max((r["share_any"] for r in mine), default=None)})
    return out


def set_totals(images):
    """Totals over the set and the best achievable result (LAB-EXPERIMENT, chain, point 6)."""
    totals = {"n_images": len(images), "n_tiles": sum(i["n_tiles"] for i in images),
              "best_achievable_tiles": sum(i["n_tiles"] for i in images),
              "best_achievable_images": sum(i["n_tiles"] > 0 for i in images)}
    for t in SELECTION:
        key = f"n_any_le_{round(t * 100)}"
        totals[key] = sum(i[key] for i in images)
        totals[f"images_with_tiles_{key}"] = sum(i[key] > 0 for i in images)
    return totals


def statement(entry):
    """The page's one-sentence claim, built from the run-file entry of the image only."""
    return (f"Снимок {entry['code']} (ортопантомограмма): {entry['n_tiles']} полных фрагментов 512 × 512; доля фона и "
            f"рамки share_any ≤ 10 % — у {entry['n_any_le_10']}, ≤ 25 % — у {entry['n_any_le_25']}, ≤ 50 % — у "
            f"{entry['n_any_le_50']}; наибольшая во фрагменте — {round(entry['max_share_any'] * 100)} %.")


STATEMENT_RE = re.compile(r"^Снимок (D\d\d) \(ортопантомограмма\): (\d+) полных фрагментов 512 × 512; доля фона и рамки "
                          r"share_any ≤ 10 % — у (\d+), ≤ 25 % — у (\d+), ≤ 50 % — у (\d+); наибольшая во фрагменте — "
                          r"(\d+) %\.$")


def check_statement(text, entry):
    """The claim against the run-file entry, field by field (G21); a mismatch is a refusal."""
    m = STATEMENT_RE.match(text)
    if not m:
        raise RunRefusal(f"{entry['code']}: the page claim does not have the declared form")
    got = (m[1], *map(int, m.groups()[1:]))
    want = (entry["code"], entry["n_tiles"], entry["n_any_le_10"], entry["n_any_le_25"], entry["n_any_le_50"],
            round(entry["max_share_any"] * 100))
    if got != want:
        raise RunRefusal(f"{entry['code']}: the page claim {got} differs from the run file {want}")


def reduced(values, scale=PAGE_SCALE):
    """Block mean of scale x scale (anti-aliased reduction) over the part divisible by scale."""
    h, w = (values.shape[0] // scale) * scale, (values.shape[1] // scale) * scale
    return values[:h, :w].reshape(h // scale, scale, w // scale, scale).mean(axis=(1, 3))


def draw_page(values, tiles, title, claim, path=None):
    """The page (and the bench view) of one image: reduced picture, grid of full tiles, share_any in %, claim and
    the four-part explanation. Returns the texts put on the figure; saves PNG to `path` if given."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    small = reduced(values)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_axes([0.02, 0.24, 0.96, 0.68])
    ax.imshow(small, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_axis_off()
    texts = [title, claim, *(f"На пальцах: {EXPLANATION[0]}" if k == 0 else EXPLANATION[k] for k in range(4))]
    for t in tiles:
        x, y, s = t["x"] / PAGE_SCALE, t["y"] / PAGE_SCALE, TILE / PAGE_SCALE
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), s, s, fill=False, edgecolor="#ffcc00", linewidth=1.5))
        label = f"{round(t['share_any'] * 100)} %"
        ax.text(x + s / 2, y + s / 2, label, color="#ffcc00", fontsize=16, ha="center", va="center",
                bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none"})
        texts.append(label)
    fig.suptitle(title, fontsize=15, y=0.97)
    fig.text(0.02, 0.19, claim, fontsize=12, wrap=True)
    for k, line in enumerate(texts[2:6]):
        fig.text(0.02, 0.15 - 0.035 * k, line, fontsize=11, wrap=True)
    if path is not None:
        fig.savefig(path, dpi=100)
    plt.close(fig)
    return texts
