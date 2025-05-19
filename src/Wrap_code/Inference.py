import os
import argparse
from pathlib import Path
from tqdm import tqdm

# Existing utility classes for PDF page splitting and conversion
from pdf_utils import PDFSplitter, PDFConverter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split one or more PDFs into pages, convert to images, and save"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_pdf", type=str,
        help="Path to a single input PDF file"
    )
    group.add_argument(
        "--input_dir", type=str,
        help="Directory containing multiple PDF files to process"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory where page images will be saved (subfolders per PDF)"
    )
    return parser.parse_args()


def process_pdf(pdf_path: Path, output_base: Path, converter: PDFConverter):
    """
    Split the given PDF into pages, convert each page to an image,
    and save images under output_base/pdf_stem/*.png
    """
    # Prepare output directory for this PDF
    pdf_output = output_base / pdf_path.stem
    pdf_output.mkdir(parents=True, exist_ok=True)

    # Instantiate splitter for this PDF file
    splitter = PDFSplitter(str(pdf_path), str(pdf_output))

    # Split PDF into single-page PDF files
    page_files = splitter.split()

    # Convert each single-page PDF to image and save
    for page_file in tqdm(page_files, desc=f"Processing {pdf_path.name}"):
        image = converter.convert(page_file)

        # Derive filename from the single-page PDF's stem
        page_stem = Path(page_file).stem
        out_path = pdf_output / f"{page_stem}.png"

        image.save(str(out_path))


def main():
    args = parse_args()
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Initialize a single converter instance (stateless)
    converter = PDFConverter()

    # Gather PDF paths
    if args.input_dir:
        pdf_paths = list(Path(args.input_dir).rglob("*.pdf"))
    else:
        pdf_paths = [Path(args.input_pdf)]

    # Process each PDF file
    for pdf_path in pdf_paths:
        if not pdf_path.is_file():
            print(f"Warning: {pdf_path} is not a file, skipping.")
            continue
        process_pdf(pdf_path, output_base, converter)

    print(f"Processed {len(pdf_paths)} PDFs. Images saved under {args.output_dir}.")


if __name__ == "__main__":
    main()
