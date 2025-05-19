"""
pdf_to_tex_cli.py

Command‑line utility that scans a folder for PDF files, runs OCR on every page
using `process_pdf_folder` (provided by `ocr.pdf_to_latex_parser`), and writes a
LaTeX file next to each source PDF.

Usage (interactive prompt):
    python pdf_to_tex_cli.py

Usage (direct argument):
    python pdf_to_tex_cli.py /path/to/pdf/folder
"""

import os
import sys
from typing import Dict

try:
    from ocr.pdf_to_latex_parser import process_pdf_folder
except ImportError as exc:
    sys.stderr.write(
        "Error: could not import process_pdf_folder from ocr.pdf_to_latex_parser.\n"
    )
    sys.stderr.write("Make sure your OCR package is installed and discoverable.\n")
    raise


def _clean_prefix(text: str) -> str:
    """Remove a leading "LaTeX:" tag if present (case‑insensitive)."""
    prefix = "latex:"
    if text.lower().startswith(prefix):
        return text[len(prefix):].strip()
    return text.strip()


def build_tex_content(pages: Dict[str, str]) -> str:
    """Construct a minimal LaTeX document from page‑level OCR results."""
    lines = [r"\documentclass{article}", r"\begin{document}"]

    for i, (_, latex_src) in enumerate(pages.items()):
        clean_src = _clean_prefix(latex_src)
        lines.append(clean_src)
        if i < len(pages) - 1:
            lines.append(r"\newpage")

    lines.append(r"\end{document}")
    return "\n".join(lines)


def main() -> None:
    # Accept a directory from argv or prompt interactively
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = input("Enter the folder path containing PDF files: ").strip()

    if not os.path.isdir(input_dir):
        sys.stderr.write(f"Error: '{input_dir}' is not a valid directory.\n")
        sys.exit(1)

    print("Transferring…")

    # Call OCR routine
    pdf_map = process_pdf_folder(input_dir)

    # Iterate over OCR results and write .tex files
    for pdf_key, pages in pdf_map.items():
        # Deduce base filename. If the key already ends with .pdf, strip it.
        base_name = pdf_key[:-4] if pdf_key.lower().endswith(".pdf") else pdf_key
        output_tex_path = os.path.join(input_dir, f"{base_name}.tex")

        tex_content = build_tex_content(pages)
        with open(output_tex_path, "w", encoding="utf‑8") as tex_file:
            tex_file.write(tex_content)

    print("Transfer completed")


if __name__ == "__main__":
    main()
