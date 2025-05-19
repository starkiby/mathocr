# ==========================================
# File: test_pdf_splitter.py
# Author: Lucille Zheng
# Date: 2025-04-26
# Description:
#   Unit tests for the PDFSplitter class.
#   Tests splitting functionality for multi-page PDFs, single-page PDFs,
#   error handling for nonexistent input, and automatic output directory creation.
# ==========================================

import os
import shutil
import unittest
from pdf_splitter import PDFSplitter

class TestPDFSplitter(unittest.TestCase):
    """
    Unit test case for the PDFSplitter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment before any test cases are run.
        Creates sample file paths and ensures output directory exists.
        """
        cls.sample_pdf = "tests/data/sample.pdf"  # Multi-page
        cls.single_page_pdf = "tests/data/single_page.pdf"  # Single-page
        cls.nonexistent_pdf = "tests/data/does_not_exist.pdf" # Nonexistent file path
        cls.output_dir = "tests/output"

        if not os.path.exists(cls.output_dir):
            os.makedirs(cls.output_dir)

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the test environment after all test cases are run.
        Removes the output directory.
        """
        if os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir)

    def test_split_multi_page_pdf(self):
        """
        Test splitting a multi-page PDF into individual pages.
        """
        splitter = PDFSplitter(self.sample_pdf, self.output_dir)
        output_files = splitter.split()
        self.assertEqual(len(output_files), len(splitter))
        for i, file in enumerate(output_files, start=1):
            expected_name = f"sample_page_{i}.pdf"
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, expected_name)))

    def test_split_single_page_pdf(self):
        """
        Test splitting a single-page PDF into one page.
        """
        splitter = PDFSplitter(self.single_page_pdf, self.output_dir)
        output_files = splitter.split()
        self.assertEqual(len(output_files), 1)
        self.assertTrue(output_files[0].endswith("single_page_page_1.pdf"))

    def test_nonexistent_input_pdf(self):
        """
        Test behavior when the input PDF file does not exist.
        Expect FileNotFoundError.
        """
        with self.assertRaises(FileNotFoundError):
            PDFSplitter(self.nonexistent_pdf, self.output_dir)

    def test_auto_create_output_dir(self):
        """
        Test that the output directory is automatically created if it does not exist.
        """
        temp_dir = "tests/temp_output"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        splitter = PDFSplitter(self.sample_pdf, temp_dir)
        splitter.split()
        self.assertTrue(os.path.exists(temp_dir))
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    unittest.main()
