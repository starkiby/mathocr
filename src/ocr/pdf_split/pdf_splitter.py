# ==========================================
# File: pdf_splitter.py
# Author: Lucille Zheng
# Date: 2025-04-22
# Description: 
#   A Python module to split a PDF file (single or multi-page) 
#   into individual single-page PDF files.
#   Designed as a feature module for a larger pipeline: 
#   PDF → Image → OCR → LaTeX Post-processing.
# ==========================================
import os
from PyPDF2 import PdfReader, PdfWriter

class PDFSplitter:
    """
    A class to split a PDF file into multiple single-page PDF files.

    Attributes:
        input_pdf_path (str): Path to the input PDF file.
        output_dir (str): Directory where split PDF pages will be saved.
        reader (PdfReader): PyPDF2 reader object for the input PDF.
        total_pages (int): Number of pages in the input PDF.
        base_name (str): Base name of the input PDF (without extension).
    """
    def __init__(self, input_pdf_path, output_dir):
        """
        Initializes the PDFSplitter with input file path and output directory.
        """
        self.input_pdf_path = input_pdf_path
        self.output_dir = output_dir
        self.reader = PdfReader(input_pdf_path)
        self.total_pages = len(self.reader.pages)
        self.base_name = os.path.splitext(os.path.basename(input_pdf_path))[0]

        # Create output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def split(self):
        """
        Splits the PDF into individual single-page PDFs.

        Returns:
            list[str]: List of output file paths for each single-page PDF.
        """
        output_files = []
        for i, page in enumerate(self.reader.pages):
            writer = PdfWriter()
            writer.add_page(page)

            output_path = os.path.join(
                self.output_dir, f"{self.base_name}_page_{i+1}.pdf"
            )
            with open(output_path, "wb") as f:
                writer.write(f)

            output_files.append(output_path)

        return output_files

    def __len__(self):
        """
        Returns the number of pages in the input PDF.
        """
        return self.total_pages
    
if __name__ == "__main__":
    # Example usage:

    # Multi pages data sample:
    input_path = "./test_data/ScannedPages2.pdf"
    output_dir = "./test_output"

    splitter = PDFSplitter(input_path, output_dir)
    output_files = splitter.split()

    expected_base = os.path.splitext(os.path.basename(input_path))[0]

    # Simple assertions to verify split result
    assert len(output_files) == len(splitter), "Mismatch between output file count and PDF page count"
    for i, file_path in enumerate(output_files, start=1):
        assert os.path.exists(file_path), f"Missing output file for page {i}"
        assert file_path.endswith(f"{expected_base}_page_{i}.pdf"), f"Incorrect file name: {file_path}"

    print(f"✅ Successfully split {len(splitter)} pages.")
