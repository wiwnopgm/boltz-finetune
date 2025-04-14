#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Boltz-1 MCP Server

This server provides tools and resources for:
- Preparing inputs for Boltz-1 architecture
- Running training and fine-tuning on Boltz-1
- Running inference with Boltz-1
- Analyzing results
"""

import os
from pathlib import Path
from typing import Optional

import yaml
import torch
from mcp.server.fastmcp import FastMCP
from mcp_tools.boltz_tools import run_command
from utils.get_slurm_template import get_slurm_script_with_command

# Initialize FastMCP server
mcp = FastMCP("Boltz-1 Server")

# === RESOURCES ===

@mcp.resource("boltz://docs")
def get_documentation() -> str:
    """Get documentation about Boltz-1 and this server."""
    return """
    # Boltz-1 Server

    This server provides tools for working with the Boltz-1 architecture for protein structure prediction.
    
    ## Available Resources
    - `boltz://docs` - This documentation
    - `boltz://config/{config_name}` - Access configuration templates
    - `boltz://model/{model_path}` - Access information about a model checkpoint
    
    ## Available Tools
    - `prepare_inputs` - Prepare inputs for Boltz-1
    - `train_model` - Run training on Boltz-1 model
    - `finetune_model` - Fine-tune a Boltz-1 model
    - `run_inference` - Run inference with Boltz-1 model
    - `analyze_results` - Analyze results from Boltz-1 predictions
    
    ## Remote Access Tools
    - `connect_ssh` - Configure SSH access to a remote cluster
    - `check_remote_path` - Check the current path on a remote server
    - `change_remote_dir` - Change directory on a remote server
    - `start_redis_server` - Start a Redis server with custom configuration
    - `start_ccd_redis_server` - Start the CCD Redis server on port 7777
    - `start_taxonomy_redis_server` - Start the Taxonomy Redis server on port 7778
    - `start_all_redis_servers` - Start both CCD and Taxonomy Redis servers
    
    ## Configuration Templates
    - `train_full` - Full model training configuration
    - `finetune_lora` - LoRA fine-tuning configuration 
    - `inference` - Inference configuration
    """


@mcp.resource("boltz://config/{config_name}")
def get_config_template(config_name: str) -> str:
    """Get a configuration template for Boltz-1.
    
    Args:
        config_name: The name of the configuration template to retrieve.
            Options: train_full, finetune_lora, inference
    """
    config_templates = {
        "train_full": """
trainer:
  accelerator: gpu
  devices: 1
  precision: 32
  gradient_clip_val: 10.0
  max_epochs: -1
  accumulate_grad_batches: 128

wandb:
  name: boltz_training_run
  project: boltz-training
  entity: your-entity

output: /path/to/output
pretrained: /path/to/pretrained/model.pth
resume: null
disable_checkpoint: false
matmul_precision: null
save_top_k: -1

data:
  datasets:
    - _target_: boltz.data.module.training.DatasetConfig
      target_dir: /path/to/processed/targets
      msa_dir: /path/to/msas
      prob: 1.0
      sampler:
        _target_: boltz.data.sample.cluster.ClusterSampler
      cropper:
        _target_: boltz.data.crop.boltz.BoltzCropper
        min_neighborhood: 0
        max_neighborhood: 40
      split: /path/to/validation_ids.txt

  # Rest of configuration omitted for brevity
  # See full_finetune.yaml for complete example
""",
        "finetune_lora": """
method: lora
rank: 8
alpha: 16
dropout: 0.0
offload_to_cpu: false

model_path: /path/to/pretrained/model
output_dir: /path/to/output

batch_size: 8
learning_rate: 1e-3
weight_decay: 0.0
num_epochs: 5
warmup_steps: 0
""",
        "inference": """
data: /path/to/input/data.fasta
out_dir: /path/to/output
cache: ~/.boltz
checkpoint: /path/to/model.ckpt
devices: 1
accelerator: gpu
recycling_steps: 3
sampling_steps: 200
diffusion_samples: 1
step_scale: 1.638
write_full_pae: false
write_full_pde: false
output_format: mmcif
num_workers: 2
override: false
seed: 42
use_msa_server: false
msa_server_url: https://api.colabfold.com
msa_pairing_strategy: greedy
"""
    }
    
    if config_name not in config_templates:
        return f"Error: Configuration template '{config_name}' not found. Available templates: {', '.join(config_templates.keys())}"
    
    return config_templates[config_name]


@mcp.resource("boltz://model/{model_path}")
def get_model_info(model_path: str) -> str:
    """Get information about a Boltz-1 model.
    
    Args:
        model_path: Path to the model checkpoint
    """
    try:
        if not Path(model_path).exists():
            return f"Error: Model checkpoint not found at {model_path}"
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Extract information
        info = {
            "model_path": model_path,
            "type": "Boltz-1 Model",
            "state_dict_keys": list(checkpoint["state_dict"].keys())[:10],  # First 10 keys
            "hyperparameters": checkpoint.get("hyper_parameters", {}),
        }
        
        return yaml.dump(info, sort_keys=False)
    except Exception as e:
        return f"Error accessing model information: {str(e)}"

@mcp.tool()
def connect_ssh(
    host: str,
    username: str,
    identity_file: str,
    check_path: bool = False,
) -> str:
    """Connect to a remote server via SSH.
    
    Args:
        host: The hostname or IP address of the remote server
        username: The username to use for SSH connection
        identity_file: Path to the SSH private key file
        check_path: Whether to check the current path on the remote server
        change_dir: Directory to change to on the remote server (optional)
        
    Returns:
        Status message with connection details
    """
    try:
        # Check if the identity file exists
        if not Path(identity_file).exists():
            return f"Error: SSH identity file not found at {identity_file}"
        
        # Create SSH config directory if it doesn't exist
        ssh_config_dir = Path.home() / ".ssh"
        ssh_config_dir.mkdir(exist_ok=True, mode=0o700)
        
        # Create or update SSH config file
        ssh_config_path = ssh_config_dir / "config"
        
        # Read existing config if it exists
        existing_config = ""
        if ssh_config_path.exists():
            with open(ssh_config_path, "r") as f:
                existing_config = f.read()
        
        # Check if the host is already configured
        host_config = f"Host {host}\n  HostName {host}\n  User {username}\n  IdentityFile {identity_file}\n  Port 22\n\n"
        
        if f"Host {host}\n" in existing_config:
            # Replace existing host config
            import re
            pattern = f"Host {host}\n.*?(?=\nHost|\Z)"
            new_config = re.sub(pattern, host_config, existing_config, flags=re.DOTALL)
        else:
            # Append new host config
            new_config = existing_config + host_config
        
        # Write updated config
        with open(ssh_config_path, "w") as f:
            f.write(new_config)
        
        # Set proper permissions for the config file
        os.chmod(ssh_config_path, 0o600)
        
        # Set proper permissions for the identity file
        os.chmod(identity_file, 0o600)
        
        # Prepare SSH command
        ssh_cmd = f"ssh {host}"
        
        # Execute SSH command if path check or directory change is requested
        if check_path:
            import subprocess
            try:
                result = subprocess.run(ssh_cmd, shell=True, check=True, capture_output=True, text=True)
                return f"SSH configuration updated for host {host}.\nRemote path: {result.stdout.strip()}"
            except subprocess.CalledProcessError as e:
                return f"SSH configuration updated for host {host}, but command failed: {e.stderr}"
        
        return f"SSH configuration updated for host {host}.\nYou can now connect using: ssh {host}"
    except Exception as e:
        return f"Error configuring SSH: {str(e)}"

@mcp.tool()
def command_remote_dir(
    command: str,
    host: str,
    directory: Optional[str] = None,
) -> str:
    """Change directory on a remote server.
    
    Args:
        host: The hostname or IP address of the remote server
        directory: Directory to change to on the remote server
        
    Returns:
        Status message with new path
    """
    try:
        import subprocess
        
        # Prepare SSH command
        if directory:
            ssh_cmd = f"ssh {host} '{command} {directory} && pwd'"
        else:
            ssh_cmd = f"ssh {host} '{command} && pwd'"
        
        # Execute SSH command
        result = subprocess.run(ssh_cmd, shell=True, check=True, capture_output=True, text=True)
        return f"Changed to directory: {result.stdout.strip()}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def interpret_command(user_intent: str) -> str:
    """Interpret user intent and return the appropriate terminal command.
    
    Args:
        user_intent: The user's intent expressed in natural language
        
    Returns:
        The appropriate terminal command
    """
    # Convert intent to lowercase for easier matching
    intent = user_intent.lower().strip()
    
    # Dictionary mapping common intents to commands
    intent_to_command = {
        "change directory": "cd",
        "list all the files in directory": "ls",
        "list files": "ls",
        "list directory": "ls",
        "show directory contents": "ls",
        "list all files": "ls",
        "list files in directory": "ls",
        "list directory contents": "ls",
        "show files": "ls",
        "show all files": "ls",
        "show all files in directory": "ls",
        "show directory": "ls",
        "show current directory": "pwd",
        "print working directory": "pwd",
        "show current path": "pwd",
        "what is my current directory": "pwd",
        "what is my current path": "pwd",
        "where am i": "pwd",
        "create directory": "mkdir",
        "make directory": "mkdir",
        "new directory": "mkdir",
        "remove directory": "rmdir",
        "delete directory": "rmdir",
        "remove file": "rm",
        "delete file": "rm",
        "copy file": "cp",
        "move file": "mv",
        "rename file": "mv",
        "show file contents": "cat",
        "display file contents": "cat",
        "print file contents": "cat",
        "read file": "cat",
        "find file": "find",
        "search for file": "find",
        "locate file": "find",
        "grep text": "grep",
        "search text": "grep",
        "find text": "grep",
        "search in files": "grep",
        "show process": "ps",
        "list processes": "ps",
        "show running processes": "ps",
        "kill process": "kill",
        "stop process": "kill",
        "terminate process": "kill",
        "show disk usage": "df",
        "show disk space": "df",
        "show directory size": "du",
        "show folder size": "du",
        "show file size": "du",
        "show memory usage": "free",
        "show memory": "free",
        "show system info": "uname -a",
        "show system information": "uname -a",
        "show os info": "uname -a",
        "show operating system": "uname -a",
        "show network": "ifconfig",
        "show network interfaces": "ifconfig",
        "show ip": "ifconfig",
        "show ip address": "ifconfig",
        "show network connections": "netstat",
        "show open ports": "netstat",
        "show listening ports": "netstat",
        "show open connections": "netstat",
        "show environment variables": "env",
        "show env": "env",
        "show path": "echo $PATH",
        "show current user": "whoami",
        "show username": "whoami",
        "show who i am": "whoami",
        "show who am i": "whoami",
        "show current user": "whoami",
        "show date": "date",
        "show time": "date",
        "show current date": "date",
        "show current time": "date",
        "show calendar": "cal",
        "show month": "cal",
        "show year": "cal",
        "clear screen": "clear",
        "clear terminal": "clear",
        "clean screen": "clear",
        "clean terminal": "clear",
        "history": "history",
        "show history": "history",
        "show command history": "history",
        "show previous commands": "history",
    }
    
    # Check if the intent matches any of our known intents
    for key, command in intent_to_command.items():
        if key in intent:
            return command
    
    # If no match is found, return a helpful message
    return "I don't recognize that intent. Please try a different command or be more specific."


@mcp.tool()
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

@mcp.tool()
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
    
@mcp.tool()
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

@mcp.tool()
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
    
if __name__ == "__main__":
    print("Starting Boltz-1 Server...")
    mcp.run()