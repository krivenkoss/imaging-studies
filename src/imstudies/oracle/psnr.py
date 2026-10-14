"""High-precision PSNR oracle for check V1-a of ../REFERENCE.md#b02.

Independent path: pure Python integers and `decimal`; no numpy, no import of
imstudies.core. Images are sequences of rows of integers 0..255.
"""

from decimal import Decimal, localcontext


def sse(ref_rows, dist_rows):
    """Sum of squared differences, exact integer."""
    total = 0
    for r_row, d_row in zip(ref_rows, dist_rows, strict=True):
        for r, d in zip(r_row, d_row, strict=True):
            total += (r - d) * (r - d)
    return total


def psnr(ref_rows, dist_rows, digits=50):
    """PSNR = 10*log10(255^2 * N / SSE) in dB as a Decimal with `digits` significant digits.

    Returns Decimal('Infinity') when SSE = 0.
    """
    n = sum(len(row) for row in ref_rows)
    e = sse(ref_rows, dist_rows)
    if e == 0:
        return Decimal("Infinity")
    with localcontext() as ctx:
        ctx.prec = digits + 10
        value = 10 * (Decimal(65025 * n) / Decimal(e)).log10()
        ctx.prec = digits
        return +value
