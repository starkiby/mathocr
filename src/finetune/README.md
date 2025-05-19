# Finetuning on VLMs

This folder includes tests of finetuning Qwen2.5-VL-7B with 1K samples. The finetuning process follows a Supervised Fine-Tuning schema.

The folder also contains a test on distilling the Qwen2.5-VL-7B to a 3B version model.


- **qwen2_5vl.ipynb**: A Jupyter Notebook that outlines the finetuning process for the Qwen2.5-VL-7B model. It includes steps for data preprocessing, model configuration, training, and evaluation.

- **distill.py**: A Python script that implements knowledge distillation techniques to enhance the finetuning process.

## Usage

1. **qwen2_5vl.ipynb**:
   - Open the notebook in Jupyter.
   - Follow the provided instructions to preprocess your dataset, configure the model, and initiate the finetuning process.
   - Evaluate the model's performance using the evaluation metrics outlined in the notebook.

2. **distill.py**:
   - Ensure all required dependencies are installed.
   - Run the script to perform knowledge distillation, which can help in creating a more efficient model by leveraging the insights from a larger pre-trained model.

## TODO
- [ ] SFT on the Qwen2.5-VL-7B with more samples, e.g. from 3k–10k samples.
- [ ] SFT on other VLMs.
- [ ] SFT VLMs on samples with multiple formulas in a single image.
- [ ] Distill the model to a smaller model, around 0.5B or even fewer parameters.