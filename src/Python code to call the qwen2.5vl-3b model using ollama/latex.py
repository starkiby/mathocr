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
    * Only output the content contained between \begin{document} and \end{document}
    * Absolutely do not include any extra output

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


def get_latex_code(image_path: str, ollama_url: str = "http://localhost:11434/api") -> str:
    """
    Convert an image file's content to LaTeX code using a local Ollama server with the qwen2.5vl:3b model.

    Parameters:
        image_path (str): The path to the image file.
        ollama_url (str, optional): The base URL of the Ollama API. Defaults to "http://localhost:11434/api".

    Returns:
        str: The LaTeX code generated from the image content.

    Raises:
        FileNotFoundError: If the image file does not exist at the specified path.
        Exception: For errors during the Ollama API call or processing.
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
                pass # Ignore errors if removal fails
            
        # Prepare the API request for Ollama
        endpoint = f"{ollama_url}/chat"
        
        payload = {
            "model": "qwen2.5vl:3b",  # The model being used in Ollama
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": " ", # You can add a specific text prompt here if needed
                    "images": [image_data] # Pass base64 image data directly
                }
            ],
            "options": {
                "temperature": 1.0,
                "top_k": 64,
                "top_p": 0.95,
                "num_predict": 8192, # Corresponds to max_tokens
            },
            "stream": False # We want a single response
        }
        
        # Send the request to Ollama
        response = requests.post(endpoint, json=payload)
        
        # Check for successful response
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
        
        result = response.json()
        if "message" in result and "content" in result["message"]:
            latex_code = result["message"]["content"]
            return latex_code
        elif "error" in result:
            raise Exception(f"Ollama API returned an error: {result['error']}")
        else:
            raise Exception(f"Ollama API call failed. Unexpected response format: {response.text}")

    # Handle potential errors during file reading or API interaction
    except FileNotFoundError as e:
        # Re-raise the FileNotFoundError to be handled by the caller
        raise e
    except requests.exceptions.RequestException as e:
        # Handle network-related errors (e.g., connection refused)
        raise Exception(f"Ollama API request failed: {e}")
    except Exception as e:
        # Catch other potential exceptions
        # Ensure the original exception message is preserved or enhanced
        error_message = str(e)
        if "Ollama API" not in error_message: # Avoid redundant phrasing
            error_message = f"Ollama API call failed or processing error: {e}"
        raise Exception(error_message)


# --- Example Usage ---
# Note: Ensure Ollama is running and the qwen2.5vl:3b model is available (e.g., run 'ollama pull qwen2.5vl:3b')

# image_file = ""  # <-- IMPORTANT: Replace with the actual path to your image file

# try:
#     latex_output = get_latex_code(image_file)
#     print("Generated LaTeX Code:")
#     print(latex_output)
# except FileNotFoundError as e:
#     print(e)
# except Exception as e: # Catching a broader exception type from the function
#     print(f"An error occurred: {e}")
# --- End Example Usage ---