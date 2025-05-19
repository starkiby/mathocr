import os
import json
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import io
from datasets import Dataset
from tqdm import tqdm
import traceback
import gc
import time
import sys


def load_hme100k(base_dir):
    """Load the HME100K dataset from pickle files and caption text files"""
    print("Loading HME100K dataset...")
    data = []

    # Load train data
    train_captions_path = os.path.join(base_dir, "train", "caption.txt")
    train_images_path = os.path.join(base_dir, "train", "images.pkl")

    if not os.path.exists(train_captions_path):
        print(f"Warning: {train_captions_path} does not exist")
    else:
        if not os.path.exists(train_images_path):
            print(f"Warning: {train_images_path} does not exist")
        else:
            with open(train_captions_path, 'r', encoding='utf-8') as f:
                train_captions = f.readlines()

            with open(train_images_path, 'rb') as f:
                train_images = pickle.load(f)

            for line in tqdm(train_captions, desc="Processing HME100K train"):
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    img_name, latex_code = parts
                    if img_name in train_images:
                        img_array = train_images[img_name]
                        img = Image.fromarray(img_array)
                        data.append({"image": img, "text": latex_code})

    # Load test data
    test_captions_path = os.path.join(base_dir, "test", "caption.txt")
    test_images_path = os.path.join(base_dir, "test", "images.pkl")

    if os.path.exists(test_captions_path) and os.path.exists(test_images_path):
        with open(test_captions_path, 'r', encoding='utf-8') as f:
            test_captions = f.readlines()

        with open(test_images_path, 'rb') as f:
            test_images = pickle.load(f)

        for line in tqdm(test_captions, desc="Processing HME100K test"):
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                img_name, latex_code = parts
                if img_name in test_images:
                    img_array = test_images[img_name]
                    img = Image.fromarray(img_array)
                    data.append({"image": img, "text": latex_code})

    print(f"Loaded {len(data)} samples from HME100K")
    return data


def load_im2latex(base_dir):
    """Load the im2latex dataset from parquet files"""
    print("Loading im2latex dataset...")
    data = []

    # Load all parquet files
    for file_name in ["train.parquet", "test.parquet", "val.parquet"]:
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            for _, row in tqdm(df.iterrows(), desc=f"Processing im2latex {file_name}", total=len(df)):
                formula = row['formula']
                img_bytes = row['image']['bytes']
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    data.append({"image": img, "text": formula})
                except Exception as e:
                    print(f"Error processing im2latex image: {e}")

    print(f"Loaded {len(data)} samples from im2latex")
    return data


def load_mlhme38k(base_dir):
    """Load the MLHME38K dataset from image folder and labels text file"""
    print("Loading MLHME38K dataset...")
    data = []

    # Load data
    train_labels_path = os.path.join(base_dir, "train_labels.txt")
    train_images_dir = os.path.join(base_dir, "train_images")

    if not os.path.exists(train_labels_path):
        print(f"Warning: {train_labels_path} does not exist")
        return data

    if not os.path.exists(train_images_dir):
        print(f"Warning: {train_images_dir} directory does not exist")
        return data

    with open(train_labels_path, 'r', encoding='utf-8') as f:
        train_labels = f.readlines()

    for line in tqdm(train_labels, desc="Processing MLHME38K"):
        parts = line.strip().split('\t', 1)
        if len(parts) == 2:
            img_name, latex_code = parts
            img_path = os.path.join(train_images_dir, img_name)
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    data.append({"image": img, "text": latex_code})
                except Exception as e:
                    print(f"Error processing MLHME38K image {img_path}: {e}")
            else:
                if len(data) < 5:  # Limit the number of warnings
                    print(f"Warning: Image file not found: {img_path}")

    print(f"Loaded {len(data)} samples from MLHME38K")
    return data


def load_m2e(base_dir):
    """Load the M2E dataset from image folder and jsonl files"""
    print("Loading M2E dataset...")
    data = []

    images_dir = os.path.join(base_dir, "images")

    if not os.path.exists(images_dir):
        print(f"Warning: Images directory {images_dir} does not exist for M2E dataset")
        # Try alternative directory structure
        images_dir = base_dir
        print(f"Trying to use base directory {images_dir} instead")

    print(f"Using images directory: {images_dir}")

    # Check if the directory has images
    if os.path.exists(images_dir) and os.path.isdir(images_dir):
        img_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if not img_files:
            print(f"Warning: No image files found in {images_dir}")
            if os.path.exists(os.path.join(base_dir, "Images")):
                images_dir = os.path.join(base_dir, "Images")
                print(f"Trying Images directory with capital I: {images_dir}")
                if os.path.isdir(images_dir):
                    img_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                    if not img_files:
                        print(f"Warning: No image files found in {images_dir} either")
        else:
            print(f"Found {len(img_files)} image files in {images_dir}")

    # Load all jsonl files
    for file_name in ["train.jsonl", "test.jsonl", "val.jsonl"]:
        jsonl_path = os.path.join(base_dir, file_name)
        if os.path.exists(jsonl_path):
            print(f"Found {file_name} at {jsonl_path}")
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            failed = 0
            for i, line in enumerate(tqdm(lines, desc=f"Processing M2E {file_name}")):
                try:
                    item = json.loads(line)
                    img_name = item['name']
                    latex_code = item['tex']
                    img_path = os.path.join(images_dir, img_name)

                    if not os.path.exists(img_path):
                        failed += 1
                        if failed <= 5:  # Limit number of error messages
                            print(f"Image not found: {img_path}")
                        continue

                    img = Image.open(img_path).convert('RGB')
                    data.append({"image": img, "text": latex_code})

                    if i < 3 and len(data) > 0:  # Print a few examples to verify
                        print(f"Successfully processed: {img_name} -> {latex_code[:30]}...")

                except Exception as e:
                    failed += 1
                    if failed <= 5:  # Limit number of error messages
                        print(f"Error processing M2E line: {e}")

            print(f"Failed to process {failed} out of {len(lines)} samples from {file_name}")

    print(f"Loaded {len(data)} total samples from M2E")
    return data


def convert_to_conversation_format_batched(dataset, batch_size=1000, output_dir="./unified_math_ocr_dataset"):
    """Convert dataset to the conversation format required for fine-tuning in batches to manage memory"""
    instruction = "Write the LaTeX representation for this image."

    # Create directory to save batches
    os.makedirs(output_dir, exist_ok=True)

    # Process data in batches
    total_samples = len(dataset)
    num_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division

    print(f"Converting {total_samples} samples to conversation format in {num_batches} batches...")

    all_dataset_pieces = []

    try:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_samples)

            print(f"\nProcessing batch {batch_idx + 1}/{num_batches} (samples {start_idx} to {end_idx - 1})...")

            batch_formatted_data = []

            # Process each sample in the current batch with a progress bar
            for sample_idx in tqdm(range(start_idx, end_idx), desc=f"Batch {batch_idx + 1}/{num_batches}"):
                sample = dataset[sample_idx]
                conversation = [
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": instruction},
                         {"type": "image", "image": sample["image"]}
                     ]
                     },
                    {"role": "assistant",
                     "content": [
                         {"type": "text", "text": sample["text"]}
                     ]
                     }
                ]
                batch_formatted_data.append({"messages": conversation})

            # Convert this batch to a Dataset object
            batch_dataset = Dataset.from_list(batch_formatted_data)

            # Report memory usage
            print(f"Batch {batch_idx + 1} converted to Dataset object with {len(batch_dataset)} samples")

            # Save this batch to disk
            batch_path = os.path.join(output_dir, f"batch_{batch_idx}")
            batch_dataset.save_to_disk(batch_path)
            print(f"Batch {batch_idx + 1} saved to {batch_path}")

            # Keep track of all batches we've saved
            all_dataset_pieces.append(batch_path)

            # Free up memory
            del batch_formatted_data
            del batch_dataset
            gc.collect()

            # Print memory status
            print(f"Memory freed after processing batch {batch_idx + 1}")

        # Now load and concatenate all batches
        print("\nLoading and concatenating all batches...")
        final_datasets = []

        for batch_path in tqdm(all_dataset_pieces, desc="Loading batches"):
            batch_dataset = Dataset.load_from_disk(batch_path)
            final_datasets.append(batch_dataset)

        # Concatenate all datasets
        from datasets import concatenate_datasets
        unified_dataset = concatenate_datasets(final_datasets)

        # Save the final unified dataset
        print(f"Saving final unified dataset with {len(unified_dataset)} samples...")
        unified_dataset.save_to_disk(output_dir)

        # Clean up batch directories
        for batch_path in all_dataset_pieces:
            import shutil
            shutil.rmtree(batch_path, ignore_errors=True)

        return unified_dataset

    except Exception as e:
        print(f"Error in convert_to_conversation_format_batched: {e}")
        traceback.print_exc()

        # Try to save what we have so far
        if all_dataset_pieces:
            print(f"Attempting to save what we have processed so far...")
            try:
                from datasets import concatenate_datasets
                processed_datasets = []

                for batch_path in all_dataset_pieces:
                    try:
                        batch_dataset = Dataset.load_from_disk(batch_path)
                        processed_datasets.append(batch_dataset)
                    except:
                        print(f"Could not load batch from {batch_path}")

                if processed_datasets:
                    partial_dataset = concatenate_datasets(processed_datasets)
                    partial_dataset.save_to_disk(output_dir)
                    print(f"Saved partial dataset with {len(partial_dataset)} samples to {output_dir}")
                    return partial_dataset
            except Exception as inner_e:
                print(f"Failed to save partial dataset: {inner_e}")

        return None


def main():
    base_dir = "/home/hh/math ocr"
    output_dir = "./unified_math_ocr_dataset"
    batch_size = 1000  # Process 1000 samples at a time - adjust based on your RAM

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load all datasets
    hme100k_dir = os.path.join(base_dir, "HME100K")
    im2latex_dir = os.path.join(base_dir, "im2latex")
    mlhme38k_dir = os.path.join(base_dir, "MLHME38K")
    m2e_dir = os.path.join(base_dir, "M2E")

    # Check if directories exist
    for d, name in [(hme100k_dir, "HME100K"), (im2latex_dir, "im2latex"),
                    (mlhme38k_dir, "MLHME38K"), (m2e_dir, "M2E")]:
        if not os.path.exists(d):
            print(f"Warning: Directory {name} not found at {d}")
        else:
            print(f"Found directory {name} at {d}")

    # Process each dataset
    all_data = []

    # Load HME100K
    hme100k_data = load_hme100k(hme100k_dir)
    all_data.extend(hme100k_data)
    print(f"Total data so far: {len(all_data)} samples")

    # Load im2latex
    im2latex_data = load_im2latex(im2latex_dir)
    all_data.extend(im2latex_data)
    print(f"Total data so far: {len(all_data)} samples")

    # Load MLHME38K
    mlhme38k_data = load_mlhme38k(mlhme38k_dir)
    all_data.extend(mlhme38k_data)
    print(f"Total data so far: {len(all_data)} samples")

    # Load M2E
    m2e_data = load_m2e(m2e_dir)
    all_data.extend(m2e_data)

    print(f"Total combined samples: {len(all_data)}")

    if len(all_data) == 0:
        print("Error: No data loaded!")
        return

    # Convert to conversation format in batches to manage memory
    unified_dataset = convert_to_conversation_format_batched(all_data, batch_size=batch_size, output_dir=output_dir)

    # Print final summary
    if unified_dataset:
        print("\nDataset integration complete!")
        print(f"Unified dataset saved to {output_dir} with {len(unified_dataset)} samples")
        print("Now you can use this dataset for fine-tuning with Qwen2-VL.")
    else:
        print("\nDataset integration failed or was incomplete.")
        print("Check the error messages above and try again with a smaller batch size if needed.")


if __name__ == "__main__":
    main()