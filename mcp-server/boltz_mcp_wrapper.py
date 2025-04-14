#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper for Boltz-1 MCP Server tools.
This provides a unified interface for accessing Boltz-1 functions using FastMCP tools.
"""

import os
import sys
import json
import pathlib
import importlib.util
from typing import Dict, List, Any, Optional, Union, Callable

# Get the directory where the current script is located
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

# Path to the boltz_mcp_server.py file
SERVER_PATH = SCRIPT_DIR / "boltz_mcp_server.py"

# Dictionary to store the imported tools
TOOLS = {}

# Import the tools from boltz_mcp_server.py
def import_server_tools():
    """Import tools from boltz_mcp_server.py."""
    global TOOLS
    
    if not SERVER_PATH.exists():
        print(f"Error: boltz_mcp_server.py not found at {SERVER_PATH}")
        return False
    
    # Create a module spec
    spec = importlib.util.spec_from_file_location("boltz_mcp_server", SERVER_PATH)
    
    # Create a module from the spec
    server_module = importlib.util.module_from_spec(spec)
    
    # Add the module to sys.modules
    sys.modules["boltz_mcp_server"] = server_module
    
    # Execute the module
    try:
        spec.loader.exec_module(server_module)
    except ImportError as e:
        print(f"Warning: Could not fully import boltz_mcp_server.py: {e}")
        return False
    
    # Extract the FastMCP tools
    # Resources
    if hasattr(server_module, "get_documentation"):
        TOOLS["get_documentation"] = server_module.get_documentation
    if hasattr(server_module, "get_config_template"):
        TOOLS["get_config_template"] = server_module.get_config_template
    if hasattr(server_module, "get_model_info"):
        TOOLS["get_model_info"] = server_module.get_model_info
    
    # Tools
    if hasattr(server_module, "prepare_inputs"):
        TOOLS["prepare_inputs"] = server_module.prepare_inputs
    if hasattr(server_module, "train_model"):
        TOOLS["train_model"] = server_module.train_model
    if hasattr(server_module, "finetune_model"):
        TOOLS["finetune_model"] = server_module.finetune_model
    if hasattr(server_module, "run_inference"):
        TOOLS["run_inference"] = server_module.run_inference
    if hasattr(server_module, "analyze_results"):
        TOOLS["analyze_results"] = server_module.analyze_results
    if hasattr(server_module, "connect_ssh"):
        TOOLS["connect_ssh"] = server_module.connect_ssh
    if hasattr(server_module, "check_remote_path"):
        TOOLS["check_remote_path"] = server_module.check_remote_path
    if hasattr(server_module, "change_remote_dir"):
        TOOLS["change_remote_dir"] = server_module.change_remote_dir
    if hasattr(server_module, "activate_environment"):
        TOOLS["activate_environment"] = server_module.activate_environment
    if hasattr(server_module, "start_redis_server"):
        TOOLS["start_redis_server"] = server_module.start_redis_server
    if hasattr(server_module, "start_ccd_redis_server"):
        TOOLS["start_ccd_redis_server"] = server_module.start_ccd_redis_server
    if hasattr(server_module, "start_taxonomy_redis_server"):
        TOOLS["start_taxonomy_redis_server"] = server_module.start_taxonomy_redis_server
    
    return True

# Import the tools
import_server_tools()

# Get a function from the imported tools
def get_tool(name: str) -> Callable:
    """Get a function from the imported tools."""
    if name in TOOLS:
        return TOOLS[name]
    else:
        # If the tool is not available, raise an error
        raise ImportError(f"Tool '{name}' not found in boltz_mcp_server.py") 