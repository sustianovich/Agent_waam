import markdown
import pdfkit

# Convert markdown to HTML
with open("output.md", "r", encoding="utf-8") as f:
    md_text = f.read()

html = markdown.markdown(md_text)

# Save as PDF
pdfkit.from_string(html, "fabricacion_aditiva_from_md.pdf")

