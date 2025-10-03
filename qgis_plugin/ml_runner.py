import subprocess
import os

def run_reclassification(input_path, model_path, output_path):
    script_path = r"ml_veg_seg_inference.py"
    cmd = [
        "python", script_path,
        "-reclass", input_path,
        "-h5", model_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr