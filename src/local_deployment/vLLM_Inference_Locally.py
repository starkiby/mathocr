import os
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
# Ensure qwen_vl_utils.py is accessible (e.g., in the same directory or Python path)
# You might need to download it from the Qwen-VL repository if you haven't already.
# https://github.com/QwenLM/Qwen-VL/blob/master/vllm_wrapper/qwen_vl_utils.py
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Error: qwen_vl_utils.py not found. Make sure it's in your Python path.")
    print("You can download it from: https://github.com/QwenLM/Qwen-VL/blob/master/vllm_wrapper/qwen_vl_utils.py")
    exit()


# --- Configuration ---
MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
# <<<--- IMPORTANT: Update this path to your actual image file
LOCAL_IMAGE_PATH = ""

# --- LaTeX Conversion Prompt ---
latex_system_prompt = """You are an AI assistant specialized in converting PDF images to LaTeX format. Please follow these instructions for the conversion:

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

# --- Check if Image Exists ---
if not os.path.exists(LOCAL_IMAGE_PATH):
    print(f"Error: Image file not found at {LOCAL_IMAGE_PATH}")
    exit()

# --- Initialize LLM ---
print(f"Loading LLM from {MODEL_PATH}...")
llm = LLM(
    model=MODEL_PATH,
    # Limit set for imageslimit is irrelevant now but kept for structure if needed later
    limit_mm_per_prompt={"image": 10},
    # Add other necessary vLLM parameters if needed (e.g., tensor_parallel_size, dtype)
    # dtype="bfloat16", # Example: uncomment if needed
)
print("LLM loaded.")

# --- Define Sampling Parameters ---
# Note: max_tokens might need to be increased significantly for long LaTeX documents
sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.9, # Slightly increased top_p might allow for more natural LaTeX structures
    repetition_penalty=1.05,
    max_tokens=1024, # Increased max_tokens for potentially long LaTeX output
)

# --- Define Input Messages for Image ---
image_messages = [
    {"role": "system", "content": latex_system_prompt},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": LOCAL_IMAGE_PATH, # Use the local image path
                # These pixel limits help the processor handle image scaling
                "min_pixels": 224 * 224,
                "max_pixels": 1280 * 28 * 28, # Example limit, adjust if necessary based on model/utils
            },
            {
                "type": "text",
                # Simple instruction, relying on the detailed system prompt
                "text": "Convert the content of the provided image to LaTeX format according to the instructions."
            },
        ],
    },
]

# --- Select Messages (Now only image messages) ---
messages = image_messages

# --- Process Input ---
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True) # Added trust_remote_code=True
print("Processor loaded.")

print("Applying chat template...")
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print("Processing vision info...")
# We only expect image_inputs here, video related outputs should be None or empty
# Still call with return_video_kwargs=True as the utility function expects it
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
print("Vision info processed.")

# --- Prepare Multi-Modal Data for LLM ---
mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs
# No need to check for video_inputs as we are only processing images

llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
    # Include mm_processor_kwargs even if empty/None, as vLLM might expect the key structure
    "mm_processor_kwargs": video_kwargs if video_kwargs else {},
}

# --- Generate Output ---
print("Generating LaTeX output...")
outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text
print("Generation complete.")

# --- Print Result ---
print("\n--- Generated LaTeX Output ---")
print(generated_text)
print("----------------------------")