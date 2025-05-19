import os
import threading
import tkinter as tk
from tkinter import filedialog
from ocr.pdf_to_latex_parser import process_pdf_folder


def launch_gui():
    # Updated data structure: PDF -> Page -> Content
    pages = {
        "PDF 1": {
            "Page 1": "Welcome to Math OCR!",
            "Page 2": "Welcome to Math OCR!"
        },
        "PDF 2": {
            "Page 1": "Welcome to Math OCR!",
            "Page 2": "Welcome to Math OCR!"
        }
    }

    pages2 = {
        "PDF 1": {
            "Page 1": "LaTeX: a^2 + b^2 = c^2 ccccccccc",
            "Page 2": "LaTeX: \\frac{1}{2} + \\frac{1}{3} = \\frac{5}{6}"
        },
        "PDF 2": {
            "Page 1": "LaTeX: \\int_0^1 x^2 dx = \\frac{1}{3}"
        }
    }

    # Global variables to track current selections
    current_pdf = None
    current_page = None
    current_path = None
    current_pages = pages

    def copy_click():
        # Get all text from the output Text widget (excluding the trailing newline)
        text = text_output.get("1.0", "end-1c")

        # Clear the system clipboard
        window.clipboard_clear()

        # Append the text to the clipboard
        window.clipboard_append(text)

        # Ensure clipboard contents are updated (especially on some platforms)
        window.update()

    def select_folder():
        nonlocal current_path
        folder_path = filedialog.askdirectory(
            title="Select a folder"
        )
        if folder_path:
            current_path = folder_path
            label_path.config(text=f"Selected:\n{folder_path}")

    def update_text():
        """Update the Text output based on current_pdf and current_page."""
        if current_pdf and current_page:
            text_output.config(state="normal")
            text_output.delete("1.0", "end")
            text_output.insert("1.0", current_pages[current_pdf][current_page])
            text_output.config(state="disabled")
            status_label.config(text=f"Viewing: {current_pdf} - {current_page}")

    def on_pdf_select(event):
        """When a PDF is selected, update the page list and set first page selected."""
        nonlocal current_pdf, current_page, current_pages

        selected_indices = pdf_listbox.curselection()
        if selected_indices:
            current_pdf = pdf_listbox.get(selected_indices[0])

            page_listbox.delete(0, "end")
            for page in current_pages[current_pdf]:
                page_listbox.insert("end", page)

            if current_pages[current_pdf]:
                # page_listbox.selection_set(0)
                current_page = page_listbox.get(0)
                update_text()

    def on_page_select(event):
        """When a page is selected, update the Text output."""
        nonlocal current_pdf, current_page

        selected_indices = page_listbox.curselection()
        if selected_indices:
            current_page = page_listbox.get(selected_indices[0])
            update_text()

    def transfer_click():
        def do_transfer():
            nonlocal current_pages, current_path, current_pdf, current_page

            if not current_path:
                label_path.config(text="Please select a folder first.")
                return

            label_path.config(text="Transferring...")

            try:
                # result = pages2
                result = process_pdf_folder(current_path)

                # Print result summary
                print("📄 OCR result summary:")
                for pdf_name, pages_dict in result.items():
                    page_count = len(pages_dict)
                    print(f"  - {pdf_name}: {page_count} page(s)")

            except Exception as e:
                label_path.config(text=f"Error: {e}")
                return

            # Update GUI
            current_pages.clear()
            current_pages.update(result)

            pdf_listbox.delete(0, "end")
            for pdf_name in current_pages:
                pdf_listbox.insert("end", pdf_name)

            if current_pages:
                current_pdf = next(iter(current_pages))
                current_page = next(iter(current_pages[current_pdf]))
                pdf_listbox.selection_set(0)
                on_pdf_select(None)
                update_text()

            label_path.config(text="Transfer complete")

        # Run in background thread
        threading.Thread(target=do_transfer).start()

    def create_tex_file():
        nonlocal current_pages

        # Ask user to select an output folder
        output_dir = filedialog.askdirectory(
            title="Select a folder to save .tex files"
        )
        if not output_dir:
            label_path.config(text="No folder selected.")
            return

        # Generate one .tex file for each PDF
        for pdf_name, pages_dict in current_pages.items():
            tex_filename = f"{pdf_name.replace(' ', '_')}.tex"
            tex_path = os.path.join(output_dir, tex_filename)

            try:
                with open(tex_path, "w", encoding="utf-8") as f:
                    # Optionally write LaTeX preamble here
                    f.write("\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n\n")

                    for page_title, latex_code in pages_dict.items():
                        f.write(f"% {page_title}\n")
                        f.write(f"\\[\n{latex_code}\n\\]\n\n")

                    f.write("\\end{document}\n")

            except Exception as e:
                label_path.config(text=f"Error saving {tex_filename}")
                print(f"[Error] {tex_filename}: {e}")

    # Create the main window
    window = tk.Tk()
    window.title("Math OCR")
    window.geometry("1080x720")

    # Create a label
    label_welcome = tk.Label(window, text="Welcome to Math OCR!", font=("Arial", 14))
    label_welcome.pack(pady=10)

    # Select
    label_path = tk.Label(window, text="Please select folder", font=("Arial", 14))
    label_path.pack(pady=10)

    # Create a button frame
    button_frame = tk.Frame(window)
    button_frame.pack(pady=10)

    # Create buttons
    button1 = tk.Button(button_frame, text="Select Folder", command=select_folder)
    button1.pack(side="left", padx=10)

    button2 = tk.Button(button_frame, text="Transfer", command=transfer_click)
    button2.pack(side="left", padx=10)

    button3 = tk.Button(button_frame, text="Copy", command=copy_click)
    button3.pack(side="left", padx=10)

    button4 = tk.Button(button_frame, text="Create .tex file", command=create_tex_file)
    button4.pack(side="left", padx=10)

    # Create an output frame
    output_frame = tk.Frame(window)
    output_frame.pack(pady=10, fill="both", expand=True)

    # Left frame: PDF listbox + scrollbar
    pdf_frame = tk.Frame(output_frame)
    pdf_frame.pack(side="left", fill="y", padx=10)

    pdf_scrollbar = tk.Scrollbar(pdf_frame)
    pdf_scrollbar.pack(side="right", fill="y")

    pdf_listbox = tk.Listbox(pdf_frame, width=20, yscrollcommand=pdf_scrollbar.set)
    for pdf in current_pages:
        pdf_listbox.insert("end", pdf)
    pdf_listbox.pack(side="left", fill="y")
    pdf_listbox.bind("<<ListboxSelect>>", on_pdf_select)

    pdf_scrollbar.config(command=pdf_listbox.yview)

    # Middle frame: Page listbox + scrollbar
    page_frame = tk.Frame(output_frame)
    page_frame.pack(side="left", fill="y", padx=10)

    page_scrollbar = tk.Scrollbar(page_frame)
    page_scrollbar.pack(side="right", fill="y")

    page_listbox = tk.Listbox(page_frame, width=20, yscrollcommand=page_scrollbar.set)
    page_listbox.pack(side="left", fill="y")
    page_listbox.bind("<<ListboxSelect>>", on_page_select)

    page_scrollbar.config(command=page_listbox.yview)

    # Right frame: text output + scrollbar
    right_frame = tk.Frame(output_frame)
    right_frame.pack(side="right", expand=True, fill="both", padx=10)

    text_scrollbar = tk.Scrollbar(right_frame)
    text_scrollbar.pack(side="right", fill="y")

    text_output = tk.Text(right_frame, wrap="word", font=("Arial", 12), yscrollcommand=text_scrollbar.set)
    text_output.pack(expand=True, fill="both")
    text_output.config(state="disabled")

    text_scrollbar.config(command=text_output.yview)

    # Bottom status bar
    status_label = tk.Label(window, text="No selection yet.", font=("Arial", 12))
    status_label.pack(pady=5)

    # Default: select the first PDF and load its first page
    pdf_listbox.selection_set(0)
    pdf_listbox.activate(0)
    pdf_listbox.focus_set()
    on_pdf_select(None)

    # Start the main event loop
    window.mainloop()
