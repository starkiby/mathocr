import os
import re
import matplotlib.pyplot as plt

def latex_to_image_mathtext(latex_formula: str, output_path: str,
                            fontsize: int = 20, dpi: int = 300):
    """
      - latex_formula: formula in latex form
      - output_path: path to save
      - fontsize: font size
      - dpi: output: higher DPI means higher resolution
    """

    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0.0)

    # wrapped in a single pair of $...$, in case
    text = fig.text(0, 0, f"${latex_formula}$", fontsize=fontsize)


    fig.canvas.draw()
    bbox = text.get_window_extent()

    width, height = bbox.size / float(dpi) + 0.02 
    fig.set_size_inches((width, height))

    # Save figure as PNG 
    fig.savefig(output_path, dpi=dpi, transparent=True,
                bbox_inches='tight', pad_inches=0.01)

    plt.close(fig)


def extract_and_render(input_path: str, output_dir: str):

    os.makedirs(output_dir, exist_ok=True)
    
    content = open(input_path, 'r', encoding='utf-8').read()

    # Use a regular expression to extract all paired '''...''' blocks (including multi-line)
    raws = re.findall(r"'''(.*?)'''", content, re.DOTALL)
    if not raws:

        print(" no blocks found.")
        return

    count = 0
    for idx, raw in enumerate(raws, 1):
        # Strip whitespace and any leading/trailing dollar signs, this part with help from AI
        cleaned = re.sub(r'^\$+|\$+$', '', raw.strip(), flags=re.DOTALL).strip()
        # Skip empty or whitespace-only
        if not cleaned:
            print(f"[{idx}] Skipped: empty or dollar-only block")
            
            continue

        count += 1
        out_file = os.path.join(output_dir, f"formula_{count}.png")
        print(f"[{idx}] Rendering → {out_file}")
        try:
            latex_to_image_mathtext(cleaned, out_file)
        except Exception as e:
            print(f"Output fail: {e}")



if __name__ == "__main__":
    # read formulas.txt，output to the same directory, change the arg if name   
    # changes.
    extract_and_render("formulas_test.txt", "out_images")
