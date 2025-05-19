import ollama
import base64


def get_latex_code(image_path: str) -> str:
    """
    Convert an image to LaTeX code using the Ollama model.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        str: The LaTeX code generated from the image.
    """
    # Define the system prompt for the model
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
       - Use LaTeX table environments (e.g., \begin{tabular} ... \end{tabular}) to format tables.
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

    with open(image_path, "rb") as f:
        image = base64.b64encode(f.read()).decode("utf-8")
    response = ollama.chat(
        model="minicpm-v",
        messages=[
            {
                'role': 'system',
                'content': SYSTEM_PROMPT,

            },
            {
                'role': 'user',
                'content': 'Convert the content of the provided image to LaTeX format according to the instructions.',
                'images': [image]
            }],
        options={
            'max_tokens': 1024,  # Limit the maximum number of tokens in the response
        }
    )
    return response["message"]["content"]

# Example usage
image_path = "I:/test/1.png"  # Replace with your file
latex_code = get_latex_code(image_path)
print(latex_code)
