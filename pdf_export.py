import io
import markdown as md
from xhtml2pdf import pisa

PDF_CSS = """
<style>
  body { font-family: Helvetica, Arial, sans-serif; font-size: 11pt; color: #222222; line-height: 1.5; }
  h1 { font-size: 19pt; color: #4C1D95; margin-bottom: 4pt; }
  h2 { font-size: 14pt; color: #5B21B6; margin-top: 16pt; margin-bottom: 4pt; }
  h3 { font-size: 12pt; color: #333333; margin-top: 12pt; margin-bottom: 3pt; }
  p { margin: 4pt 0; }
  ul, ol { margin-left: 16pt; }
  li { margin: 2pt 0; }
  code { background: #f0f0f0; padding: 1pt 3pt; font-family: Courier, monospace; }
  pre { background: #f5f5f5; padding: 8pt; }
  hr { border: none; border-top: 1px solid #ddd; margin: 12pt 0; }
</style>
"""


def markdown_to_pdf_bytes(title: str, markdown_text: str) -> bytes:
    """Render a title + markdown body into a styled PDF, returned as raw bytes."""
    body_html = md.markdown(markdown_text, extensions=["fenced_code", "tables"])
    html = f"<html><head>{PDF_CSS}</head><body><h1>{title}</h1>{body_html}</body></html>"

    buffer = io.BytesIO()
    pisa.CreatePDF(src=io.StringIO(html), dest=buffer)
    return buffer.getvalue()
