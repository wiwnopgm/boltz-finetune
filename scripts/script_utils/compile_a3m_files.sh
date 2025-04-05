#!/bin/bash

# Set source and destination directories
SOURCE_DIR="/ist-nas/users/bunditb/boltz/scripts/examples/colabfold_output"
DEST_DIR="/ist-nas/users/bunditb/boltz/scripts/examples/compiled_a3m_files"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

echo "Searching for a3m files in $SOURCE_DIR"

# Count the number of a3m files
A3M_FILES=($(find "$SOURCE_DIR" -name "*.a3m"))
TOTAL_FILES=${#A3M_FILES[@]}
echo "Found $TOTAL_FILES a3m files to process"

# Counter for successful copies
COPIED=0

# Process each a3m file
for a3m_file in "${A3M_FILES[@]}"; do
    # Get the base filename
    base_name=$(basename "$a3m_file")
    
    # Remove the "mpro-" prefix if it exists
    new_name="${base_name#mpro-}"
    
    # Copy to destination with the new name
    cp "$a3m_file" "$DEST_DIR/$new_name"
    
    # Check if copy was successful
    if [ $? -eq 0 ]; then
        ((COPIED++))
        echo "Copied: $base_name → $new_name"
    else
        echo "Failed to copy: $a3m_file"
    fi
done

echo "Successfully compiled $COPIED/$TOTAL_FILES a3m files"
echo "Files saved to: $DEST_DIR"

# List the contents of the destination directory
echo "Contents of destination directory:"
ls -la "$DEST_DIR" 