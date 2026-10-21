"""Executor of B12 in the separate environment (pillow-heif 0.22.0): AVIF through libheif, encode and decode.

Contract: ../REFERENCE.md#b12. Run by imstudies.lab.coders.avif_libheif as `python -I <this file> <work dir>`
with the python of the separate environment; imports only the standard library, numpy, PIL and pillow_heif of
that environment, never imstudies. Reads <work>/request.json (qualities, enc_params, options) and
<work>/image.npy; writes q<qqq>.avif and q<qqq>_recon.npy for every quality and result.json; prints the
versions of the environment as one JSON line on every call.
"""

import io
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)  # pillow-heif 0.22.0 marks AVIF deprecated

import numpy as np  # noqa: E402
import PIL  # noqa: E402
import pillow_heif  # noqa: E402
from PIL import Image  # noqa: E402


def _versions():
    info = pillow_heif.libheif_info()
    return {"python": sys.version.split()[0], "pillow_heif": pillow_heif.__version__, "pillow": PIL.__version__,
            "numpy": np.__version__, "libheif": info.get("libheif"), "aom_encoder": info.get("encoders", {}).get("aom"),
            "aom_decoder": info.get("decoders", {}).get("aom")}, info


def _main(work):
    request = json.loads((work / "request.json").read_text(encoding="utf-8"))
    image = np.load(work / "image.npy")
    for key, value in request["options"].items():
        setattr(pillow_heif.options, key, value)
    versions, info = _versions()
    print(json.dumps(versions), flush=True)
    items = []
    for q in request["qualities"]:
        buf = io.BytesIO()
        pillow_heif.from_pillow(Image.fromarray(image, "L")).save(buf, format="AVIF", quality=q,
                                                                  enc_params=dict(request["enc_params"]))
        data = buf.getvalue()
        heif = pillow_heif.open_heif(io.BytesIO(data), convert_hdr_to_8bit=False)
        recon = np.asarray(heif[0])
        (work / f"q{q:03d}.avif").write_bytes(data)
        np.save(work / f"q{q:03d}_recon.npy", recon)
        items.append({"quality": q, "mode": heif[0].mode, "shape": list(recon.shape), "dtype": str(recon.dtype)})
    options = {k: getattr(pillow_heif.options, k) for k in dir(pillow_heif.options) if k.isupper()}
    (work / "result.json").write_text(json.dumps({"versions": versions, "libheif_info": info, "options": options,
                                                  "items": items}, default=repr), encoding="utf-8")


if __name__ == "__main__":
    _main(Path(sys.argv[1]))
