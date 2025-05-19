import requests
import base64
import os
import json
from typing import Optional
from PIL import Image
import math

# Define the system prompt for the model
SYSTEM_PROMPT = r"""
# You are an AI assistant specialized in converting images containing text, handwritten notes, and mathematical/physical formulas into LaTeX format. Your primary goal is to produce clean, compilable LaTeX code suitable for direct insertion between \begin{document} and \end{document} in an Overleaf project.

Please adhere strictly to the following instructions:

1.  **Content Recognition and Conversion:**
    * Accurately recognize all textual content, including printed and handwritten elements, from the input image. Do not guess or infer missing or illegible content; transcribe only what is clearly discernible.
    * Convert all recognized text into standard LaTeX.
    * Preserve the original document structure, including headings, paragraphs, lists, and an accurate representation of the spatial layout of text and formulas.
    * Maintain all original line breaks precisely as they appear in the image.

2.  **Mathematical and Physical Formula Processing:**
    * Identify and convert all mathematical and physical formulas into accurate LaTeX.
    * Enclose inline formulas with `$`...`$`. For example: `This is an inline formula $E = mc^2$.`
    * Enclose block or display formulas with `$$`...`$$` or an appropriate LaTeX environment such as `\begin{equation}`...`\end{equation}` or `\begin{align}`...`\end{align}` if the structure suggests it (e.g., for multi-line equations). Ensure correct alignment and structure for complex formulas. For example: `$$ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} $$`
    * Pay close attention to subscripts, superscripts, fractions, integrals, Greek letters, and other special mathematical symbols, ensuring their correct LaTeX representation.

3.  **Table Processing:**
    * Convert any tables into compilable LaTeX table environments (e.g., `\begin{tabular}{...} ... \end{tabular}`, `\begin{array}{...} ... \end{array}`).
    * Preserve the original table structure, including the number of rows and columns, cell content, alignment (left, center, right), and any horizontal or vertical lines.
    * Ensure line breaks within cells are handled appropriately, potentially using `\parbox` or similar commands if necessary for multi-line cell content.

4.  **Figure and Image Handling:**
    * You are to ignore any purely graphical elements, diagrams, or figures present in the image that are not text, formulas, or tables. Do not attempt to describe, caption, or convert these graphical elements into LaTeX (e.g., do not generate `\includegraphics` commands).

5.  **Output Format and Quality:**
    * The output must be solely LaTeX code. Do not include any explanatory text, comments, or any content other than the LaTeX representation of the image's textual and formulaic content.
    * The output must NOT include `\documentclass{...}`, `\usepackage{...}`, `\begin{document}`, or `\end{document}`. The generated LaTeX will be placed directly inside an existing LaTeX document body.
    * Ensure the generated LaTeX is well-formatted, clean, and adheres to standard LaTeX practices to maximize compatibility and readability in Overleaf.
    * Maintain a clear structure with appropriate line breaks between LaTeX elements (e.g., paragraphs, equations, list items) to reflect the original image layout.
    * Strive for a LaTeX representation that, when compiled, visually resembles the input image's text and formula layout as closely as possible.

Strictly follow these guidelines to ensure the accuracy, completeness, and usability of the converted LaTeX code. Before finalizing the output, internally verify that the LaTeX code is syntactically correct and likely to compile without errors.
"""

def preprocess_image(image_path: str) -> str:
    """
    Preprocess the image to ensure it meets size requirements.
    
    Parameters:
        image_path (str): Path to the original image file
        
    Returns:
        str: Path to the processed image (either original or resized)
    """
    # Constants defined by requirements
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    block_size = 28
    
    # Open and get image dimensions
    image = Image.open(image_path)
    width, height = image.size
    total_pixels = width * height
    
    # Check if resizing is needed
    if min_pixels <= total_pixels <= max_pixels:
        # Check if dimensions need rounding to multiples of 28
        new_width = round(width / block_size) * block_size
        new_height = round(height / block_size) * block_size
        
        if new_width != width or new_height != height:
            image = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        # Calculate scaling factor
        if total_pixels < min_pixels:
            scale = math.sqrt(min_pixels / total_pixels)
        else:  # total_pixels > max_pixels
            scale = math.sqrt(max_pixels / total_pixels)
            
        # Calculate new dimensions as multiples of 28
        resized_width = round(width * scale / block_size) * block_size
        resized_height = round(height * scale / block_size) * block_size
        
        # Resize the image
        image = image.resize((resized_width, resized_height), Image.LANCZOS)
    
    # Save processed image to a temporary file
    temp_path = f"{os.path.splitext(image_path)[0]}_processed.jpg"
    image.save(temp_path, quality=95)
    
    return temp_path


def get_latex_code(image_path: str, lm_studio_url: str = "http://localhost:6234/v1") -> str:
    """
    Convert an image file's content to LaTeX code using local LM Studio with Qwen2.5-VL-3B-Instruct-GGUF model.

    Parameters:
        image_path (str): The path to the image file.
        lm_studio_url (str, optional): The URL of the LM Studio server. Defaults to "http://localhost:6234/v1".

    Returns:
        str: The LaTeX code generated from the image content.

    Raises:
        FileNotFoundError: If the image file does not exist at the specified path.
        Exception: For errors during the LM Studio API call or processing.
    """
    
    
    # Check if the image file exists before attempting to open it
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: The file '{image_path}' was not found.")

    try:
        # Preprocess the image to ensure proper dimensions
        processed_image_path = preprocess_image(image_path)
        # Read the processed image file in binary mode
        with open(processed_image_path, "rb") as f:
            # Encode the image data to base64
            image_data = base64.b64encode(f.read()).decode("utf-8")
        # Clean up temporary file if it's different from the original
        if processed_image_path != image_path:
            try:
                os.remove(processed_image_path)
            except:
                pass
            
        # Prepare the API request for LM Studio
        endpoint = f"{lm_studio_url}/chat/completions"
        
        payload = {
            "model": "unsloth/Qwen2.5-VL-3B-Instruct-GGUF",  # The model being used in LM Studio
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": " "
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.95,
            "min_p": 0.0,
            "max_tokens": 2048,
        }
        
        # Send the request to LM Studio
        response = requests.post(endpoint, json=payload)
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            latex_code = result["choices"][0]["message"]["content"]
            return latex_code
        else:
            raise Exception(f"LM Studio API call failed with status code {response.status_code}: {response.text}")

    # Handle potential errors during file reading or API interaction
    except FileNotFoundError as e:
        # Re-raise the FileNotFoundError to be handled by the caller
        raise e
    except Exception as e:
        # Catch other potential exceptions (network, LM Studio errors, etc.)
        print(f"An error occurred during processing: {e}")
        raise Exception(f"LM Studio API call failed or processing error: {e}")


# --- Example Usage ---
# Note: Ensure LM Studio is running with the Qwen2.5-VL-3B-Instruct-GGUF model loaded

# image_file = ""  # <-- IMPORTANT: Replace with the actual path to your image file

# try:
#     latex_output = get_latex_code(image_file)
#     print("Generated LaTeX Code:")
#     print(latex_output)
# except FileNotFoundError as e:
#     print(e)
# except ConnectionError as e:
#     print(f"Connection Error: {e}")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")
# --- End Example Usage ---