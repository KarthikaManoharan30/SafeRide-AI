
# reporter.py
from fpdf import FPDF
from PIL import Image as PILImage
import textwrap
from pathlib import Path

# reporter.py
def _wrap(val, width=88):
    s = str(val)
    # Remove zero-width hacks; rely on hard wrapping for long URLs
    return "\n".join(
        textwrap.wrap(s, width=width, break_long_words=True, break_on_hyphens=True)
    )

def _place_image(pdf: FPDF, image_path: str):
    p = Path(image_path)
    if not p.exists():
        return
    max_w = pdf.w - pdf.l_margin - pdf.r_margin
    with PILImage.open(image_path) as im:
        w, h = im.size
    pdf.image(str(p), w=max_w)

def build_incident_pdf(out_path: str, title: str, details: dict, image_path: str | None = None):
    pdf = FPDF(unit="pt", format="A4")
    pdf.set_margins(36, 36, 36)               # 0.5" margins
    pdf.set_auto_page_break(auto=True, margin=36)
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 24, title, ln=1)

    pdf.set_font("Helvetica", size=11)
    usable_w= pdf.w - pdf.l_margin - pdf.r_margin
    for k, v in details.items():
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 16, f"{k}: {_wrap(v)}")
        
    pdf.ln(8)

    if image_path:
        _place_image(pdf, image_path)

    pdf.output(out_path)
    return out_path