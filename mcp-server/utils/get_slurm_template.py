from typing import List
import os

def get_slurm_template(template_path: str):
    """
    Returns a template for a SLURM job script for training with multiple GPUs.
    Reads the template from the slurm_template.sbatch file.
    
    Returns:
        str: A string containing the SLURM script template
    """
    # Read the template file
    with open(template_path, 'r') as f:
        return f.read()
    
def write_slurm_script(template, output_path):
    """
    Writes a SLURM template to a .sbatch file.
    
    Args:
        template (str): The SLURM template string
        output_path (str): Path where the .sbatch file should be written
        
    Returns:
        str: The path to the written file
    """
    # Ensure the output path has the .sbatch extension
    if not output_path.endswith('.sbatch'):
        output_path += '.sbatch'
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write the template to the file
    with open(output_path, 'w') as f:
        f.write(template)
    
    # Make the file executable
    return output_path

def get_slurm_script_with_command(template_path: str, commands: List[str], output_path: str):
    """
    Returns a SLURM template with a custom command added at the bottom.
    
    Args:
        command (str): The command to add to the template
        
    Returns:
        str: The SLURM template with the command added
    """
    template = get_slurm_template(template_path)
    slurm_script = template + "\n\n" + "\n\n".join(commands)
    write_slurm_script(slurm_script, output_path)