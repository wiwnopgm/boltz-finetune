# === TOOLS ===

from typing import Optional
import os
import time
from pathlib import Path
from utils.get_slurm_template import (
    get_slurm_script_with_command,
)

import yaml
import subprocess

# Create a function to run commands via SSH
def run_command(
    command: str,
    remote_host: Optional[str] = None, 
    env_path: Optional[str] = None,
):
    if remote_host:
        if env_path:
            ssh_cmd = f"ssh {remote_host} 'conda activate {env_path} && {command}'"
        else:
            ssh_cmd = f"ssh {remote_host} '{command}'"
        return subprocess.run(ssh_cmd, shell=True, check=True, capture_output=True, text=True)
    else:
        return subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

def process_train_data(
    script_dir: str,
    pdb_data_dir: str,
    msa_data_dir: str,
    output_dir: str,
    ccd_port: int,
    taxonomy_port: int,
    env_path: Optional[str] = None,
    remote_host: Optional[str] = None,
) -> str:
    """Prepare all necessary inputs for Boltz-1 architecture.
    
    Args:
        script_dir: Path to the directory containing processing scripts
        pdb_data_dir: Path to directory containing PDB data
        msa_data_dir: Path to directory containing MSA data
        output_dir: Path to output directory
        redis_host: Host for Redis server
        ccd_port: Port for CCD Redis server
        taxonomy_port: Port for taxonomy Redis server
        max_seqs: Maximum number of sequences to process in MSA
        remote_host: SSH host to run the commands on
        
    Returns:
        Status message
    """

    
    # Convert all string paths to Path objects for local path handling
    script_dir = Path(script_dir)
    output_dir = Path(output_dir)
    pdb_data_dir = Path(pdb_data_dir)
    msa_data_dir = Path(msa_data_dir)
    
    # Track progress
    progress_log = []
    
    # Create remote directories
    progress_log.append(f"Creating directories on remote host {remote_host}...")
    mkdir_cmd = f"mkdir -p {output_dir} {output_dir}/processed_structures {output_dir}/processed_msa"
    run_command(mkdir_cmd, remote_host, env_path)
    
    # Check if the input directories exist on remote
    check_pdb_cmd = f"[ -d {pdb_data_dir} ] && echo 'Directory exists' || echo 'Directory not found'"
    pdb_check = run_command(check_pdb_cmd, remote_host, env_path)
    if "not found" in pdb_check.stdout:
        return f"Error: PDB data path does not exist on remote: {pdb_data_dir}"
    
    check_msa_cmd = f"[ -d {msa_data_dir} ] && echo 'Directory exists' || echo 'Directory not found'"
    msa_check = run_command(check_msa_cmd, remote_host, env_path)
    if "not found" in msa_check.stdout:
        return f"Error: MSA data path does not exist on remote: {msa_data_dir}"
    
    # Download necessary data on remote
    progress_log.append("1. Downloading necessary data on remote...")
    cache_dir = "~/.boltz"
    if remote_host:
        check_cache_cmd = f"[ -d {cache_dir} ] && echo 'Directory exists' || echo 'Directory not found'"
        cache_check = run_command(check_cache_cmd, remote_host, env_path)
        if "not found" in cache_check.stdout:
            # Directory doesn't exist, so download
            download_cmd = f"python -c 'from boltz.main import download; download(\"{cache_dir}\")'"
            run_command(download_cmd, remote_host, env_path)
            progress_log.append("✓ Cache download completed")
    else:
        # For local execution, check if directory exists
        local_cache_dir = os.path.expanduser(cache_dir)
        if not os.path.exists(local_cache_dir):
            download_cmd = f"python -c 'from boltz.main import download; download(\"{cache_dir}\")'"
            run_command(download_cmd)
            progress_log.append("✓ Cache download completed")
    
    # Process structures using CCD Redis server
    progress_log.append("2. Processing structures using CCD Redis server...")
    rcsb_cmd = (
        f"conda activate {env_path} && python {script_dir}/rcsb.py --datadir {pdb_data_dir} --outdir {output_dir}/processed_structures --redis-port {ccd_port}"
    )
    run_command(rcsb_cmd, remote_host, env_path)
    progress_log.append("✓ Structure processing completed")
    
    # Process MSA using Taxonomy Redis server
    progress_log.append("3. Processing MSA using Taxonomy Redis server...")

    msa_cmd = f"conda activate {env_path} && python {script_dir}/msa.py --msadir {msa_data_dir} --outdir {output_dir}/processed_msa --redis-port {taxonomy_port}"
    run_command(msa_cmd, remote_host, env_path)
    progress_log.append("✓ MSA processing completed")
    
    # Get file count to include in final status
    file_count_cmd = f"find {output_dir}/processed_structures {output_dir}/processed_msa -type f | wc -l"
    file_count = run_command(file_count_cmd, remote_host, env_path).stdout.strip()
    
    # Final status message
    progress_log.append("\nInput preparation completed on remote host:")
    progress_log.append(f"- Processed structures saved to: {output_dir}/processed_structures")
    progress_log.append(f"- Processed MSA files saved to: {output_dir}/processed_msa")
    progress_log.append(f"- Total files generated: {file_count}")
    
    return "\n".join(progress_log)

def run_training(
    script_dir: str,
    config_path: str,
    env_path: Optional[str] = None,
    remote_host: Optional[str] = None,
    slurm_template: Optional[str] = None,
) -> str:       
    # Load the configuration
    if not Path(config_path).exists():
        return f"Error: Configuration file not found: {config_path}"
    
    if not Path(script_dir).exists():
        return f"Error: Script directory not found: {script_dir}"

    train_cmd = f"python scripts/train/train.py {config_path}"
    if slurm_template:
        slurm_output_path = f"train_{time.time()}.sbatch"
        slurm_script = get_slurm_script_with_command(slurm_template, [train_cmd], slurm_output_path)
        slurm_cmd = f"sbatch {slurm_script}"
        run_command(slurm_cmd, remote_host, env_path)
        return f"Successfully started training with configuration: {config_path} with slurm script: {slurm_output_path}"

    else:
        run_command(train_cmd, remote_host, env_path)
        return f"Successfully started training with configuration: {config_path}"
    
def run_inference(
    config_path: str,
    env_path: Optional[str] = None,
    remote_host: Optional[str] = None,
    slurm_template: Optional[str] = None,
) -> str:       

    # Load the configuration
    if not Path(config_path).exists():
        return f"Error: Configuration file not found: {config_path}"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['inference']

    # List of optional parameters to check for in config
    optional_params = {
        'out_dir': '--out_dir',
        'custom_msa_dir': '--custom_msa_dir',
        'recycling_steps': '--recycling_steps',
        'diffusion_samples': '--diffusion_samples',
        'devices': '--devices',
        'accelerator': '--accelerator',
        'sampling_steps': '--sampling_steps',
        'step_scale': '--step_scale',
        'write_full_pae': '--write_full_pae',
        'write_full_pde': '--write_full_pde',
        'output_format': '--output_format',
        'num_workers': '--num_workers',
        'seed': '--seed',
        'use_msa_server': '--use_msa_server',
        'msa_server_url': '--msa_server_url',
        'msa_pairing_strategy': '--msa_pairing_strategy'
    }
    
    inference_cmd = f"""
     boltz predict {config['inference_config_path']}
    """

    for param, flag in optional_params.items():
        if param in config:
            inference_cmd += f" {flag} {config[param]}"

    if slurm_template:
        slurm_output_path = f"inference_{time.time()}.sbatch"
        slurm_script = get_slurm_script_with_command(slurm_template, [inference_cmd], slurm_output_path)
        slurm_cmd = f"sbatch {slurm_script}"
        run_command(slurm_cmd, remote_host, env_path)
        return f"Successfully started inference with configuration: {config_path} with slurm script: {slurm_output_path}"

    else:
        run_command(inference_cmd, remote_host, env_path)
        return f"Successfully started inference with configuration: {config_path}"

def get_prediction_results(
    results_dir: str
) -> str:
    """Analyze results from Boltz-1 predictions.
    
    Args:
        results_dir: Path to directory containing results
        output_format: Format for output (text or html)
        
    Returns:
        Analysis results
    """
    try:
        results_path = Path(results_dir)
        if not results_path.exists():
            return f"Error: Results directory not found: {results_path}"
        
        # Check for prediction directory
        pred_dir = results_path / "predictions"
        if not pred_dir.exists():
            return f"Error: Predictions directory not found in {results_path}"
        # Find all prediction directories
        pred_dirs = list(pred_dir.glob("*"))
        if not pred_dirs:
            return f"No prediction directories found in {pred_dir}"
        
        # Analyze predictions
        analysis = []
        
        for dir_path in pred_dirs:
            if not dir_path.is_dir():
                continue
                
            # Check for PDB files
            pdb_files = list(dir_path.glob("*.pdb"))
            mmcif_files = list(dir_path.glob("*.cif"))
            
            structures = pdb_files + mmcif_files
            
            # Check for confidence files
            pae_files = list(dir_path.glob("*_pae.npz"))
            plddt_files = list(dir_path.glob("*_plddt.npz"))
            
            analysis.append({
                "name": dir_path.name,
                "structures": [f.name for f in structures],
                "pae_files": [f.name for f in pae_files],
                "plddt_files": [f.name for f in plddt_files],
            })

        # Format results
        result = "Analysis of Boltz-1 Predictions\n\n"
            
        for pred in analysis:
            result += f"Target: {pred['name']}\n"
            result += f"  Structures: {len(pred['structures'])}\n"
            if pred['structures']:
                result += f"    - {', '.join(pred['structures'][:3])}"
                if len(pred['structures']) > 3:
                    result += f" and {len(pred['structures']) - 3} more"
                result += "\n"
            
        result += f"  PAE files: {len(pred['pae_files'])}\n"
        result += f"  pLDDT files: {len(pred['plddt_files'])}\n\n"
        return result
    except Exception as e:
        return f"Error analyzing results: {str(e)}"