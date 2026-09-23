"""Generate a 16x16 ORCID logo PNG for the paper's author block.

No network access: the icon is drawn directly (green circle + white ring +
white 'iD' hint) and saved to Reference/figures/orcid_16x16.png.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

SIZE = 16
CX, CY, R = 7.5, 7.5, 7.5
ORCID_GREEN = (0xA6, 0xCE, 0x39)
WHITE = (0xFF, 0xFF, 0xFF)


def inside_circle(x: float, y: float, cx: float, cy: float, r: float) -> bool:
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def pixel_color(px: int, py: int) -> tuple[int, int, int, int]:
    # Sample the pixel center.
    x, y = px + 0.5, py + 0.5
    if not inside_circle(x, y, CX, CY, R):
        return (0, 0, 0, 0)  # transparent outside the logo

    # White ring: outer radius R, inner radius R-2.
    if inside_circle(x, y, CX, CY, R - 2.0):
        # Interior: green with a white 'i' stem and dot on the left side.
        # 'i' stem: x in [4.2, 5.8], y in [5.5, 11.5]; dot above it.
        if 4.2 <= x <= 5.8 and 5.5 <= y <= 11.5:
            return (*WHITE, 255)
        if 4.2 <= x <= 5.8 and 3.0 <= y <= 4.6:
            return (*WHITE, 255)
        return (*ORCID_GREEN, 255)
    return (*WHITE, 255)  # ring


def png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def main() -> None:
    rows = []
    for py in range(SIZE):
        row = bytearray([0])  # filter type 0 (none)
        for px in range(SIZE):
            r, g, b, a = pixel_color(px, py)
            row += bytes([r, g, b, a])
        rows.append(bytes(row))

    ihdr = struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + png_chunk(b"IEND", b"")
    )

    out = Path(__file__).resolve().parent.parent / "Reference" / "figures" / "orcid_16x16.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)
    print(f"wrote {out} ({len(png)} bytes)")


if __name__ == "__main__":
    main()
