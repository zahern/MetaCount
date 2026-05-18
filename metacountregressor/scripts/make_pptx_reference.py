"""make_pptx_reference.py

Generate pptx_reference.pptx with MetaCount teal/orange colour theme.

Run once from the scripts/ directory before `quarto render`:
    python scripts/make_pptx_reference.py

The resulting pptx_reference.pptx is referenced by
washington_cmf_presentation.qmd via:
    format:
      pptx:
        reference-doc: pptx_reference.pptx

Quarto/Pandoc inherits the theme colours and table style from the reference
doc, so markdown tables automatically get a teal header row.
"""
import io
import re
import zipfile
from pathlib import Path

from pptx import Presentation

HERE = Path(__file__).parent
OUT  = HERE / "pptx_reference.pptx"

# ── MetaCount brand colours ────────────────────────────────────────────────
TEAL   = "0A6C74"   # accent1  → table header, slide-title bar underline
ORANGE = "D96F32"   # accent2  → headings, callout highlights
NAVY   = "152238"   # dk2      → primary body text / title text
# ──────────────────────────────────────────────────────────────────────────


def _patch_theme(xml_bytes: bytes) -> bytes:
    """Replace accent1 / accent2 / dk2 srgbClr values in a theme XML part."""
    text = xml_bytes.decode("utf-8")
    colours = {
        "accent1": TEAL,
        "accent2": ORANGE,
        "dk2":     NAVY,
    }
    for tag, hex_val in colours.items():
        # Self-closing form:  <a:srgbClr val="XXXXXX"/>
        text = re.sub(
            rf'(<a:{tag}[^>]*>\s*<a:srgbClr val=")[^"]*("[\s]*/?>)',
            rf'\g<1>{hex_val}\2',
            text,
            flags=re.DOTALL,
        )
        # Open/close form:  <a:srgbClr val="XXXXXX">...</a:srgbClr>
        text = re.sub(
            rf'(<a:{tag}[^>]*>[\s\S]*?<a:srgbClr val=")[^"]*(")',
            rf'\g<1>{hex_val}\2',
            text,
            flags=re.DOTALL,
        )
        # sysClr with lastClr attr (used for dk1/lt1/dk2 system colours)
        text = re.sub(
            rf'(<a:{tag}[^>]*>[\s\S]*?<a:sysClr[^>]*lastClr=")[^"]*(")',
            rf'\g<1>{hex_val}\2',
            text,
            flags=re.DOTALL,
        )
    return text.encode("utf-8")


def main() -> None:
    # 1. Create a minimal valid .pptx with python-pptx
    prs = Presentation()
    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)

    # 2. Patch every theme XML part in-memory
    out_buf = io.BytesIO()
    with zipfile.ZipFile(buf, "r") as z_in:
        with zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED) as z_out:
            for name in z_in.namelist():
                data = z_in.read(name)
                if re.match(r"ppt/theme/theme\d+\.xml$", name):
                    data = _patch_theme(data)
                z_out.writestr(name, data)

    OUT.write_bytes(out_buf.getvalue())
    print(f"Written:  {OUT}")
    print(f"  accent1 (table header / title bar) → #{TEAL}")
    print(f"  accent2 (section headings)         → #{ORANGE}")
    print(f"  dk2     (body text)                → #{NAVY}")


if __name__ == "__main__":
    main()
