import base64
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
# input the image path
image_path = ""
system_prompt = """You are an AI assistant specialized in converting PDF images to LaTeX format. Please follow these instructions for the conversion:

1. **Text Processing:**
   - Accurately recognize all text content in the PDF image without guessing or inferring.
   - Convert the recognized text into LaTeX format.
   - Maintain the original document structure, including headings, paragraphs, lists, etc.
   - Preserve all original line breaks as they appear in the PDF image.
2. **Mathematical Formula Processing:**
   - Convert all mathematical formulas to LaTeX format.
   - Enclose inline formulas with $ and $. For example: This is an inline formula $E = mc^2$.
   - Enclose block formulas with $$ and $$. For example: $$\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$
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

Please strictly adhere to these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the entire PDF image into corresponding LaTeX format without adding any extra explanations or comments."""

user_prompt_text = "Convert the content of the provided image to LaTeX format according to the instructions."
# Read the image file and encode it in base64
with open(image_path, "rb") as f:
    encoded_image = base64.b64encode(f.read())
encoded_image_text = encoded_image.decode("utf-8")
base64_qwen = f"data:image;base64,{encoded_image_text}"
# Send the image to the Qwen model for processing
chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    messages=[
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_qwen
                    },
                },
                {"type": "text", "text": user_prompt_text},
            ],
        },
    ],
)
print("Chat response:", chat_response)