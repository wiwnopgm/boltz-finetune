#!/bin/bash

# Source directory containing subdirectories with structures
SRC_DIR="/ist-nas/users/bunditb/boltz/scripts/examples/parsed_pdb_output"

# Target directory where files will be compiled
TARGET_DIR="/ist-nas/users/bunditb/boltz/scripts/examples/training_example/processed_pdb_structure"

# Create target directory and subdirectories if they don't exist
mkdir -p "$TARGET_DIR/records"
mkdir -p "$TARGET_DIR/structure"

# Count for progress reporting
total_files=0
processed_files=0

# First, count the total number of .npz and .json files in records and structures directories
total_files=$(find "$SRC_DIR" -path "*/structures/*.npz" -o -path "*/records/*.npz" -o -path "*/records/*.json" | wc -l)
echo "Found $total_files files to process"

# Process structure files
echo "Processing structure files..."
find "$SRC_DIR" -type d -name "structures" | while read struct_dir; do
    # Find all .npz files in this structure directory
    find "$struct_dir" -type f -name "*.npz" | while read file; do
        # Get the basename of the file
        filename=$(basename "$file")
        
        # Extract the file extension (.npz)
        extension="${filename##*.}"
        
        # Extract the base name without extension
        base="${filename%.*}"
        
        # Convert to lowercase before any _A or _B suffix
        if [[ "$base" =~ (.*)(_[AB])$ ]]; then
            # Get the part before the suffix
            prefix="${BASH_REMATCH[1]}"
            # Get the suffix (_A or _B)
            suffix="${BASH_REMATCH[2]}"
            
            # Convert the prefix to lowercase
            prefix_lower=$(echo "$prefix" | tr '[:upper:]' '[:lower:]')
            
            # Create the new filename with lowercase prefix and original suffix
            new_base="${prefix_lower}${suffix}"
        else
            # If there's no _A or _B suffix, just convert the whole base to lowercase
            new_base=$(echo "$base" | tr '[:upper:]' '[:lower:]')
        fi
        
        # Add back the extension
        new_filename="${new_base}.${extension}"
        
        # Copy the file to the structure subdirectory with the new name
        cp "$file" "$TARGET_DIR/structure/$new_filename"
        
        # Update processed files count
        processed_files=$((processed_files + 1))
        
        # Print progress
        echo "[$processed_files/$total_files] Copied $filename to $TARGET_DIR/structure/$new_filename"
    done
done

# Process record files (.npz and .json)
echo "Processing record files..."
find "$SRC_DIR" -type d -name "records" | while read record_dir; do
    # Find all .npz and .json files in this record directory
    find "$record_dir" -type f \( -name "*.npz" -o -name "*.json" \) | while read file; do
        # Get the basename of the file
        filename=$(basename "$file")
        
        # Extract the file extension
        extension="${filename##*.}"
        
        # Extract the base name without extension
        base="${filename%.*}"
        
        # Convert to lowercase before any _A or _B suffix
        if [[ "$base" =~ (.*)(_[AB])$ ]]; then
            # Get the part before the suffix
            prefix="${BASH_REMATCH[1]}"
            # Get the suffix (_A or _B)
            suffix="${BASH_REMATCH[2]}"
            
            # Convert the prefix to lowercase
            prefix_lower=$(echo "$prefix" | tr '[:upper:]' '[:lower:]')
            
            # Create the new filename with lowercase prefix and original suffix
            new_base="${prefix_lower}${suffix}"
        else
            # If there's no _A or _B suffix, just convert the whole base to lowercase
            new_base=$(echo "$base" | tr '[:upper:]' '[:lower:]')
        fi
        
        # Add back the extension
        new_filename="${new_base}.${extension}"
        
        # Copy the file to the records subdirectory with the new name
        cp "$file" "$TARGET_DIR/records/$new_filename"
        
        # Update processed files count
        processed_files=$((processed_files + 1))
        
        # Print progress
        echo "[$processed_files/$total_files] Copied $filename to $TARGET_DIR/records/$new_filename"
    done
done

echo "Compilation complete. All files are now available in:"
echo "  - Structure files: $TARGET_DIR/structure"
echo "  - Record files: $TARGET_DIR/records (both .npz and .json files)" 