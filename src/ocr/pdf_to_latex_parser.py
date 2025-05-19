import os
import fitz  # PyMuPDF
import tempfile
from typing import Dict
from ocr.latex import get_latex_code


def process_pdf_folder(input_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Processes all PDF files in the given folder.
    Converts each page to an image, performs LaTeX recognition,
    and stores the result in a nested dictionary.

    Parameters:
        input_dir (str): Path to the folder containing PDF files.

    Returns:
        Dict[str, Dict[str, str]]: A nested dictionary with structure:
            {
                "PDF Name": {
                    "Page 1": "LaTeX: ...",
                    "Page 2": "LaTeX: ..."
                },
                ...
            }
    """
    pages = {}
    input_dir = os.path.abspath(input_dir)

    # Create a temporary directory to store intermediate image files
    with tempfile.TemporaryDirectory() as temp_dir:
        for filename in os.listdir(input_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(input_dir, filename)
                doc = fitz.open(pdf_path)
                pdf_name = os.path.splitext(filename)[0]
                pages[pdf_name] = {}

                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=200)

                    # Save image to temporary directory
                    image_filename = f"{pdf_name}_page_{page_num + 1}.png"
                    image_path = os.path.join(temp_dir, image_filename)
                    pix.save(image_path)

                    # Use the LaTeX recognition function
                    latex_code = get_latex_code(image_path)
                    # latex_code = "11111"

                    # Store result in nested dictionary
                    page_label = f"Page {page_num + 1}"
                    pages[pdf_name][page_label] = latex_code

    return pages
