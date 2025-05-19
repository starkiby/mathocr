## Deploying Qwen2.5-VL-3B with vLLM 

This guide outlines the steps to set up your environment and deploy the `Qwen/Qwen2.5-VL-3B-Instruct` model locally using the vLLM server.

**1. Environment Setup (WSL Way)**

- **Install WSL:** If you're on Windows, it's highly recommended to use WSL (Windows Subsystem for Linux). Follow the official [Microsoft WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install). Ensure you are using WSL2.

  ```
  # Open PowerShell as Administrator
  wsl --install
  # Restart your computer if prompted
  # Verify WSL version (should show VERSION 2)
  wsl -l -v
  ```

- **Update Linux Distribution:** Open your WSL terminal (e.g., Ubuntu) and update packages:

  ```
  sudo apt update && sudo apt upgrade 
  ```

- **Install Python:** Ensure you have Python 3.10 or newer.

  ```
  sudo apt install python3 python3-pip python3-venv -y
  python3 --version
  pip3 --version
  ```

  

**2. Project Setup**

- **Create Project Directory:**

  ```
  mkdir ~/qwen_vllm_deploy
  cd ~/qwen_vllm_deploy
  ```

- **Create and Activate Virtual Environment:**

  ```
  python3 -m venv .venv
  source .venv/bin/activate
  # Your terminal prompt should now start with (.venv)
  ```

**3. Install Dependencies**

- **Install vLLM and Qwen Dependencies:**

  ```
  # Ensure you have the latest NVIDIA drivers installed on your host machine
  # Install PyTorch matching your CUDA version (check https://pytorch.org/)
  # Example for CUDA 12.1:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  
  # Install vLLM and necessary libraries
  pip install "vllm>=0.7.2" "transformers>=4.49.0" accelerate qwen-vl-utils sentencepiece protobuf Pillow requests gradio tiktoken einops
  ```

  *Note: Check the* [*vLLM installation guide*](https://docs.vllm.ai/en/latest/getting_started/installation.html) *for specific requirements based on your hardware and CUDA version.*

**4. Download Model from Hugging Face**

- **Install Git LFS:** You'll need Git Large File Storage to download the model weights.

  ```
  sudo apt-get update
  sudo apt-get install git-lfs
  git lfs install
  ```

- **Install Hugging Face CLI (Optional but Recommended):**

  ```
  pip install -U "huggingface_hub[cli]"
  ```

- **Download the Model:** Choose *one* of the following methods:

  The **current** model download **method** is to download the official Qwen2.5-VL-3B. We will upload our fine-tuned models to Hugging Face later and provide relevant links for download.

  - **Method A: Using `huggingface-cli` (Recommended)**

    ```
    # Login to Hugging Face (optional, needed for gated models, but not Qwen2.5-VL)
    # huggingface-cli login
    
    # Create a directory for the model
    mkdir models
    # Download the model (this might take a while)
    huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./models/Qwen2.5-VL-3B-Instruct --local-dir-use-symlinks False
    ```

  - **Method B: Using `git clone`**

    ```
    # Create a directory for the model
    mkdir models
    cd models
    # Clone the model repository (this might take a while)
    git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
    cd .. # Go back to the project root
    ```

**5. Start the vLLM Server**

- **Run the Server:** Navigate back to your project's root directory (`~/qwen_vllm_deploy`) if you aren't already there.

  ```
  # Ensure your virtual environment is active: source .venv/bin/activate
  
  # Start the server, pointing to the downloaded model directory
  vllm serve ./models/Qwen2.5-VL-3B-Instruct \
      --host 0.0.0.0 \
      --port 8000 \
      --limit-mm-per-prompt image=5 \
      --max-model-len 8000 \
      --dtype auto \
      --trust-remote-code
      # Add --tensor-parallel-size N if you have multiple GPUs (N=number of GPUs)
      # Use --gpu-memory-utilization 0.9 to limit GPU memory usage if needed
  ```

  - `./models/Qwen2.5-VL-3B-Instruct`: Path to the model directory you created.
  - `--limit-mm-per-prompt image=5`: Allows up to 5 images per prompt (adjust as needed).
  - `--max-model-len 8000`: Maximum context length.
  - `--dtype auto`: Automatically selects the data type (e.g., bfloat16, float16). Use `bfloat16` or `float16` explicitly if needed.
  - `--trust-remote-code`: Necessary for models like Qwen that require custom code execution.

Your vLLM server should now be running and accessible at `http://<your-ip-address>:8000`. You can use `http://localhost:8000` if accessing from the same machine where the server is running.

Then please check  Openai_Api_Service_Method.py and vLLM_Inference_Locally.py to run Qwen2.5-VL-3B on your local computer