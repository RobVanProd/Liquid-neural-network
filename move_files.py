import shutil
import os

source_files = ['liquid_s4.py', 'cfc_model.py', 'visualization.py']
target_dir = 'liquid_neural_network'

for file in source_files:
    if os.path.exists(file):
        shutil.move(file, os.path.join(target_dir, file))
        print(f"Moved {file} to {target_dir}")
