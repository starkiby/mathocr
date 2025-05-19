import ollama
import base64
import os

# Define the system prompt for the model (unchanged as requested)
SYSTEM_PROMPT = """
You are an AI assistant specialized in converting PDF images to LaTeX format. Please follow these instructions for the conversion:

1. **Text Processing:**
   - Accurately recognize all text content in the PDF image without guessing or inferring.
   - Convert the recognized text into LaTeX format.
   - Maintain the original document structure, including headings, paragraphs, lists, etc.
   - Preserve all original line breaks as they appear in the PDF image.
2. **Mathematical Formula Processing:**
   - Convert all mathematical formulas to LaTeX format.
   - Enclose inline formulas with $and$. For example: This is an inline formula $E = mc^2$.
   - Enclose block formulas with $and$. For example: $\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$.
3. **Table Processing:**
   - Convert tables into LaTeX format.
   - Use LaTeX table environments (e.g., \\begin{tabular} ... \\end{tabular}) to format tables.
   - Ensure the table structure and alignment are preserved, including proper line breaks.
4. **Figure Handling:**
   - Ignore figures in the PDF image. Do not attempt to describe or convert images.
5. **Output Format:**
   - Ensure the output document is in proper LaTeX format.
   - Maintain a clear structure with appropriate line breaks between elements.
   - For complex layouts, preserve the original document's structure and formatting as closely as possible.

Please strictly adhere to these guidelines to ensure accuracy and consistency in the conversion.
Your task is to accurately convert the content of the entire PDF image into corresponding LaTeX format without adding any extra explanations or comments.
And ensure that the final returned result does not contain any extra messages.
Before outputting the message, please ensure that the LaTeX code can compile properly.
"""

def get_latex_code(image_path: str) -> str:
    """
    Convert an image file's content to LaTeX code using the Ollama 'minicpm-v' model.

    Parameters:
        image_path (str): The path to the image file.

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
        # Read the image file in binary mode
        with open(image_path, "rb") as f:
            # Encode the image data to base64 and decode it to a UTF-8 string
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        # Send the request to the Ollama API
        response = ollama.chat(
            model="minicpm-v", # Specify the model to use
            messages=[
                {
                    'role': 'system',
                    'content': SYSTEM_PROMPT, # Provide the detailed system prompt
                },
                {
                    'role': 'user',
                    # User instruction for the conversion task
                    'content': 'Convert the content of the provided image to LaTeX format according to the instructions.',
                    # Include the base64 encoded image data
                    'images': [image_base64]
                }
            ],
            options={
                'max_tokens': 2048,  # Limit the maximum number of tokens in the response
            }
        )

        # Extract the LaTeX content from the response message
        # Assumes the model follows the prompt and only returns LaTeX
        latex_code = response["message"]["content"]
        return latex_code

    # Handle potential errors during file reading or API interaction
    except FileNotFoundError as e:
         # Re-raise the FileNotFoundError to be handled by the caller
        raise e
    except Exception as e:
        # Catch other potential exceptions (e.g., network issues, Ollama errors)
        print(f"An error occurred during processing: {e}")
        # Re-raise the exception or return an error indicator as needed
        raise Exception(f"Ollama API call failed or processing error: {e}")


# --- Example Usage ---
# Note: Ensure Ollama is running and the 'minicpm-v' model is available.

# image_file = "path/to/your/image.png" # <-- IMPORTANT: Replace with the actual path to your image file

# try:
#     latex_output = get_latex_code(image_file)
#     print("Generated LaTeX Code:")
#     print(latex_output)
# except FileNotFoundError as e:
#     print(e)
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")
# --- End Example Usage ---
