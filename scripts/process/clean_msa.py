#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import re
import shutil
import tempfile
from tqdm import tqdm

def clean_a3m_file(input_path, output_path=None):
    """
    Clean an a3m file by removing null characters and other problematic characters.
    If output_path is None, the file will be modified in place.
    """
    # Create a temp file if we're modifying in place
    if output_path is None:
        temp_fd, temp_path = tempfile.mkstemp()
        output_path = temp_path
        in_place = True
    else:
        in_place = False
    
    try:
        with open(input_path, 'rb') as f_in:
            content = f_in.read()
        
        # Replace null characters with spaces
        clean_content = content.replace(b'\x00', b' ')
        
        with open(output_path, 'wb') as f_out:
            f_out.write(clean_content)
        
        # If modifying in place, replace the original file
        if in_place:
            shutil.move(output_path, input_path)
            os.close(temp_fd)
            
        return True
    except Exception as e:
        print(f"Error cleaning {input_path}: {e}")
        if in_place and 'temp_fd' in locals():
            os.close(temp_fd)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        return False

def main():
    parser = argparse.ArgumentParser(description="Clean a3m files by removing problematic characters")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing a3m files to clean")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save cleaned files (if not specified, files will be modified in place)")
    parser.add_argument("--exclude", type=str, default="*_env*",
                        help="Pattern to exclude (default: '*_env*')")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Create output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all a3m files to process
    a3m_files = []
    for path in input_dir.rglob("*.a3m"):
        # Skip files matching the exclude pattern
        if args.exclude and re.search(args.exclude.replace("*", ".*"), str(path)):
            continue
        a3m_files.append(path)
    
    print(f"Found {len(a3m_files)} a3m files to clean")
    
    # Process each file
    cleaned_count = 0
    for a3m_path in tqdm(a3m_files):
        if output_dir:
            # Keep the relative path structure
            rel_path = a3m_path.relative_to(input_dir)
            out_path = output_dir / rel_path
            # Create parent directories if they don't exist
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = None
        
        success = clean_a3m_file(a3m_path, out_path)
        if success:
            cleaned_count += 1
    
    print(f"Successfully cleaned {cleaned_count}/{len(a3m_files)} files")

if __name__ == "__main__":
    main() 