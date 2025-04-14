#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit application for Boltz-1 MCP Server.
This provides a web interface for interacting with Boltz-1 functions using FastMCP tools.
"""

import os
import sys
import json
import pathlib
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from dotenv import load_dotenv

import streamlit as st
import yaml
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Get the directory where the current script is located
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

# Import the wrapper
from boltz_mcp_wrapper import get_tool
st.success("Successfully imported Boltz-1 MCP tools from wrapper")

# Import gemini wrapper for chat
from google import genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
from google.genai import types

def display_tool_result(result):
    """Display the result of a tool call in the Streamlit UI."""
    st.markdown(result)

async def process_function_call(session, function_call, message_placeholder, chat_history):
    """Process a function call from Gemini and handle the response."""
    try:
        # Process tool arguments
        tool_args = function_call.args
        with st.status(f"Executing tool: {function_call.name}...", expanded=True):
            st.write(f"Arguments: {tool_args}")
            
            # Check if the function is a resource (handled directly from boltz_mcp_server)
            resource_functions = ["get_documentation", "get_config_template", "get_model_info"]
            
            if function_call.name in resource_functions:
                # Import the resources directly from boltz_mcp_server
                from boltz_mcp_server import get_documentation, get_config_template, get_model_info
                
                # Call the resource function with the appropriate arguments
                if function_call.name == "get_documentation":
                    result = get_documentation()
                elif function_call.name == "get_config_template":
                    result = get_config_template(tool_args.get("config_name", "train_full"))
                elif function_call.name == "get_model_info":
                    result = get_model_info(tool_args.get("model_path", ""))
                else:
                    result = f"Resource function {function_call.name} is not properly handled"
            else:
                # Regular MCP tools are called through the session
                result = await session.call_tool(function_call.name, tool_args)
            
            st.write(f"Result: {result}")
        
        # Update chat history
        chat_history.append({"role": "model", "content": f"I'll call the {function_call.name} tool."})
        chat_history.append({"role": "function", "name": function_call.name, "content": result})
        
        return result
    except Exception as e:
        error_msg = f"Error executing tool: {str(e)}"
        st.error(error_msg)
        chat_history.append({"role": "function", "name": function_call.name, "content": error_msg})
        return error_msg

def clean_schema_for_gemini(schema):
    """Clean the schema to be compatible with Gemini's expected format."""
    if not schema:
        return {}
    
    cleaned_schema = json.loads(json.dumps(schema))
    if 'properties' in cleaned_schema:
        for prop_value in cleaned_schema['properties'].values():
            if 'default' in prop_value:
                del prop_value['default']
    
    return cleaned_schema

def main():
    """Main function to run the Streamlit app."""
    st.title("Boltz-1 MCP Tools")
    
    st.markdown("""
    This application provides access to Boltz-1 protein structure prediction tools.
    Select a tool category from the sidebar to get started.
    """)
    
    # Sidebar for tool selection
    st.sidebar.title("Tool Categories")
    
    # Tool categories
    category = st.sidebar.radio(
        "Select a category:",
        ["Resources", "Input Preparation", "Training & Fine-tuning", "Inference", "Analysis", "Remote Access", "Redis Servers", "Chat with Gemini"]
    )
    
    # Resources
    if category == "Input Preparation":
        st.header("Input Preparation")
        
        input_data_path = st.text_input("Input data path (FASTA, PDB, or CIF file):")
        output_dir = st.text_input("Output directory:")
        compute_msa = st.checkbox("Compute MSA", value=True)
        
        if st.button("Prepare Inputs"):
            if input_data_path and output_dir:
                with st.spinner("Preparing inputs..."):
                    result = get_tool("prepare_inputs")(
                        input_data_path=input_data_path,
                        output_dir=output_dir,
                        compute_msa=compute_msa
                    )
                    display_tool_result(result)
            else:
                st.error("Please provide both input data path and output directory.")
    
    # Training & Fine-tuning
    elif category == "Training & Fine-tuning":
        st.header("Training & Fine-tuning")
        
        operation = st.radio(
            "Select operation:",
            ["Train Model", "Fine-tune Model"]
        )
        
        if operation == "Train Model":
            st.subheader("Train Model")
            
            config_path = st.text_input("Configuration file path:")
            devices = st.number_input("Number of GPU devices:", min_value=1, value=1)
            max_epochs = st.number_input("Maximum epochs:", min_value=1, value=10)
            batch_size = st.number_input("Batch size:", min_value=1, value=1)
            learning_rate = st.number_input("Learning rate:", min_value=0.0001, value=0.0018, format="%.4f")
            
            if st.button("Train Model"):
                if config_path:
                    with st.spinner("Training model..."):
                        result = get_tool("train_model")(
                            config_path=config_path,
                            devices=devices,
                            max_epochs=max_epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate
                        )
                        display_tool_result(result)
                else:
                    st.error("Please provide a configuration file path.")
            
        elif operation == "Fine-tune Model":
            st.subheader("Fine-tune Model")
            
            method = st.selectbox("Fine-tuning method:", ["lora", "full"])
            model_path = st.text_input("Pre-trained model path:")
            output_dir = st.text_input("Output directory:")
            rank = st.number_input("LoRA rank:", min_value=1, value=8)
            alpha = st.number_input("LoRA alpha:", min_value=1.0, value=16.0, format="%.1f")
            dropout = st.number_input("Dropout probability:", min_value=0.0, max_value=1.0, value=0.0, format="%.2f")
            batch_size = st.number_input("Batch size:", min_value=1, value=8)
            learning_rate = st.number_input("Learning rate:", min_value=0.0001, value=0.001, format="%.4f")
            num_epochs = st.number_input("Number of epochs:", min_value=1, value=5)
            
            if st.button("Fine-tune Model"):
                if model_path and output_dir:
                    with st.spinner("Preparing for fine-tuning..."):
                        result = get_tool("finetune_model")(
                            method=method,
                            model_path=model_path,
                            output_dir=output_dir,
                            rank=rank,
                            alpha=alpha,
                            dropout=dropout,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            num_epochs=num_epochs
                        )
                        display_tool_result(result)
                else:
                    st.error("Please provide both model path and output directory.")
    
    # Inference
    elif category == "Inference":
        st.header("Inference")
        
        input_path = st.text_input("Input path (FASTA or YAML file):")
        out_dir = st.text_input("Output directory:")
        checkpoint = st.text_input("Model checkpoint (optional):")
        devices = st.number_input("Number of GPU devices:", min_value=1, value=1)
        recycling_steps = st.number_input("Recycling steps:", min_value=1, value=3)
        sampling_steps = st.number_input("Sampling steps:", min_value=1, value=200)
        diffusion_samples = st.number_input("Diffusion samples:", min_value=1, value=1)
        step_scale = st.number_input("Step scale:", min_value=0.1, value=1.638, format="%.3f")
        use_msa_server = st.checkbox("Use MSA server", value=False)
        
        if st.button("Run Inference"):
            if input_path and out_dir:
                with st.spinner("Running inference..."):
                    result = get_tool("run_inference")(
                        input_path=input_path,
                        out_dir=out_dir,
                        checkpoint=checkpoint if checkpoint else None,
                        devices=devices,
                        recycling_steps=recycling_steps,
                        sampling_steps=sampling_steps,
                        diffusion_samples=diffusion_samples,
                        step_scale=step_scale,
                        use_msa_server=use_msa_server
                    )
                    display_tool_result(result)
            else:
                st.error("Please provide both input path and output directory.")
    
    # Analysis
    elif category == "Analysis":
        st.header("Analysis")
        
        results_dir = st.text_input("Results directory:")
        output_format = st.selectbox("Output format:", ["text", "html"])
        
        if st.button("Analyze Results"):
            if results_dir:
                with st.spinner("Analyzing results..."):
                    result = get_tool("analyze_results")(
                        results_dir=results_dir,
                        output_format=output_format
                    )
                    display_tool_result(result)
            else:
                st.error("Please provide a results directory.")
    
    # Remote Access
    elif category == "Remote Access":
        st.header("Remote Access")
        
        operation = st.radio(
            "Select operation:",
            ["Connect SSH", "Check Remote Path", "Change Remote Directory", "Activate Environment"]
        )
        
        if operation == "Connect SSH":
            st.subheader("Connect SSH")
            
            host = st.text_input("Host:")
            username = st.text_input("Username:")
            identity_file = st.text_input("SSH identity file path:")
            port = st.number_input("SSH port:", min_value=1, value=22)
            check_path = st.checkbox("Check path after connection", value=False)
            change_dir = st.text_input("Directory to change to (optional):")
            
            if st.button("Connect SSH"):
                if host and username and identity_file:
                    with st.spinner("Connecting to SSH..."):
                        result = get_tool("connect_ssh")(
                            host=host,
                            username=username,
                            identity_file=identity_file,
                            port=port,
                            check_path=check_path,
                            change_dir=change_dir if change_dir else None
                        )
                        display_tool_result(result)
                else:
                    st.error("Please provide host, username, and identity file.")
            
        elif operation == "Check Remote Path":
            st.subheader("Check Remote Path")
            
            host = st.text_input("Host:")
            path = st.text_input("Path to check (optional):")
            
            if st.button("Check Remote Path"):
                if host:
                    with st.spinner("Checking remote path..."):
                        result = get_tool("check_remote_path")(
                            host=host,
                            path=path if path else None
                        )
                        display_tool_result(result)
                else:
                    st.error("Please provide a host.")
            
        elif operation == "Change Remote Directory":
            st.subheader("Change Remote Directory")
            
            host = st.text_input("Host:")
            directory = st.text_input("Directory to change to:")
            
            if st.button("Change Remote Directory"):
                if host and directory:
                    with st.spinner("Changing remote directory..."):
                        result = get_tool("change_remote_dir")(
                            host=host,
                            directory=directory
                        )
                        display_tool_result(result)
                else:
                    st.error("Please provide both host and directory.")
            
        elif operation == "Activate Environment":
            st.subheader("Activate Environment")
            
            env_path = st.text_input("Environment path:")
            host = st.text_input("Remote host (optional):")
            username = st.text_input("Username (required if host is provided):")
            identity_file = st.text_input("SSH identity file (required if host is provided):")
            
            if st.button("Activate Environment"):
                if env_path:
                    with st.spinner("Activating environment..."):
                        result = get_tool("activate_environment")(
                            env_path=env_path,
                            host=host if host else None,
                            username=username if username else None,
                            identity_file=identity_file if identity_file else None
                        )
                        display_tool_result(result)
                else:
                    st.error("Please provide an environment path.")
    
    # Redis Servers
    elif category == "Redis Servers":
        st.header("Redis Servers")
        
        operation = st.radio(
            "Select operation:",
            ["Start Custom Redis Server", "Start CCD Redis Server", "Start Taxonomy Redis Server"]
        )
        
        if operation == "Start Custom Redis Server":
            st.subheader("Start Custom Redis Server")
            
            dbfilename = st.text_input("Database filename:", value="redis.rdb")
            port = st.number_input("Port:", min_value=1, value=6379)
            host = st.text_input("Remote host (optional):")
            username = st.text_input("Username (required if host is provided):")
            identity_file = st.text_input("SSH identity file (required if host is provided):")
            env_path = st.text_input("Environment path (optional):")
            wait_for_connection = st.checkbox("Wait for connection", value=True)
            timeout_seconds = st.number_input("Timeout (seconds):", min_value=1, value=30)
            
            if st.button("Start Redis Server"):
                with st.spinner("Starting Redis server..."):
                    result = get_tool("start_redis_server")(
                        dbfilename=dbfilename,
                        port=port,
                        host=host if host else None,
                        username=username if username else None,
                        identity_file=identity_file if identity_file else None,
                        env_path=env_path if env_path else None,
                        wait_for_connection=wait_for_connection,
                        timeout_seconds=timeout_seconds
                    )
                    display_tool_result(result)
            
        elif operation == "Start CCD Redis Server":
            st.subheader("Start CCD Redis Server")
            
            wait_for_connection = st.checkbox("Wait for connection", value=True)
            timeout_seconds = st.number_input("Timeout (seconds):", min_value=1, value=30)
            
            if st.button("Start CCD Redis Server"):
                with st.spinner("Starting CCD Redis server..."):
                    result = get_tool("start_ccd_redis_server")(
                        wait_for_connection=wait_for_connection,
                        timeout_seconds=timeout_seconds
                    )
                    display_tool_result(result)
            
        elif operation == "Start Taxonomy Redis Server":
            st.subheader("Start Taxonomy Redis Server")
            
            wait_for_connection = st.checkbox("Wait for connection", value=True)
            timeout_seconds = st.number_input("Timeout (seconds):", min_value=1, value=30)
            
            if st.button("Start Taxonomy Redis Server"):
                with st.spinner("Starting Taxonomy Redis server..."):
                    result = get_tool("start_taxonomy_redis_server")(
                        wait_for_connection=wait_for_connection,
                        timeout_seconds=timeout_seconds
                    )
                    display_tool_result(result)
    
    # Chat with Gemini
    elif category == "Chat with Gemini":
        st.header("Chat with Gemini")
        
        # Initialize session state for chat
        if "gemini_messages" not in st.session_state:
            st.session_state.gemini_messages = []
        
        if "gemini_initialized" not in st.session_state:
            st.session_state.gemini_initialized = False
            st.session_state.gemini_tools = []
            st.session_state.gemini_client = None
            st.session_state.gemini_session = None
            
        # Gemini API key
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        if not gemini_api_key:
            st.error("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")
            return
        
        # Initialize Gemini and MCP session if not already initialized
        if not st.session_state.gemini_initialized:
            with st.spinner("Initializing Gemini and MCP session..."):
                try:
                    # Initialize Gemini client
                    st.session_state.gemini_client = genai.Client(api_key=gemini_api_key)
                    
                    # Set up MCP server
                    server_path = SCRIPT_DIR / "boltz_mcp_server.py"
                    server_params = StdioServerParameters(
                        command="python",
                        args=[str(server_path)],
                        env={},
                    )
                    
                    # This needs to run in an async context
                    async def init_mcp():
                        # Connect to MCP server
                        read, write = await stdio_client(server_params).__aenter__()
                        session = await ClientSession(read, write).__aenter__()
                        await session.initialize()
                        
                        # Get available tools
                        mcp_tools = await session.list_tools()
                        
                        # Create function declarations from tools
                        function_declarations = [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": clean_schema_for_gemini(tool.inputSchema),
                            }
                            for tool in mcp_tools.tools
                        ]
                        
                        # Add resource functions manually as they're not included in the tools list
                        # Add get_documentation as a tool
                        function_declarations.append({
                            "name": "get_documentation",
                            "description": "Get documentation about Boltz-1 and this server.",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                        })
                        
                        # Add get_config_template as a tool
                        function_declarations.append({
                            "name": "get_config_template",
                            "description": "Get a configuration template for Boltz-1.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "config_name": {
                                        "type": "string",
                                        "description": "The name of the configuration template to retrieve. Options: train_full, finetune_lora, inference"
                                    }
                                },
                                "required": ["config_name"]
                            }
                        })
                        
                        # Add get_model_info as a tool
                        function_declarations.append({
                            "name": "get_model_info",
                            "description": "Get information about a Boltz-1 model.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "model_path": {
                                        "type": "string",
                                        "description": "Path to the model checkpoint"
                                    }
                                },
                                "required": ["model_path"]
                            }
                        })
                        
                        return session, function_declarations
                    
                    # Run the async initialization
                    session, function_declarations = asyncio.run(init_mcp())
                    
                    # Store in session state
                    st.session_state.gemini_session = session
                    st.session_state.gemini_tools = function_declarations
                    st.session_state.gemini_initialized = True
                    
                    st.success("Gemini and MCP session initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing Gemini and MCP session: {str(e)}")
                    return
        
        # Display chat messages
        for message in st.session_state.gemini_messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.chat_message("user").write(content)
            elif role == "model":
                st.chat_message("assistant").write(content)
            elif role == "function":
                st.chat_message("assistant").write(f"Tool result: {content}")
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to chat
            st.session_state.gemini_messages.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            
            # Create Gemini request
            try:
                # Format the messages for Gemini
                contents = []
                for msg in st.session_state.gemini_messages:
                    if msg["role"] == "user":
                        contents.append(types.Content(
                            role="user",
                            parts=[types.Part(text=msg["content"])]
                        ))
                    elif msg["role"] == "model":
                        contents.append(types.Content(
                            role="model", 
                            parts=[types.Part(text=msg["content"])]
                        ))
                    elif msg["role"] == "function":
                        # For function messages, add them as user messages with tool results
                        contents.append(types.Content(
                            role="user",
                            parts=[types.Part(text=f"Tool {msg.get('name', 'unknown')} returned: {msg['content']}")]
                        ))
                
                # Create tools for Gemini
                tools = types.Tool(function_declarations=st.session_state.gemini_tools)
                
                # Display a message placeholder for the assistant's response
                message_placeholder = st.chat_message("assistant").empty()
                
                # Send request to Gemini
                async def call_gemini():
                    # Call Gemini API
                    response = await asyncio.to_thread(
                        st.session_state.gemini_client.models.generate_content,
                        model="gemini-2.0-flash",
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=0.7,
                            tools=[tools],
                        ),
                    )
                    
                    # Check for function call
                    if (response.candidates and 
                        response.candidates[0].content and 
                        response.candidates[0].content.parts and 
                        hasattr(response.candidates[0].content.parts[0], 'function_call') and 
                        response.candidates[0].content.parts[0].function_call):
                        
                        function_call = response.candidates[0].content.parts[0].function_call
                        
                        # Call the tool
                        await process_function_call(
                            st.session_state.gemini_session,
                            function_call,
                            message_placeholder,
                            st.session_state.gemini_messages
                        )
                        
                        # Generate follow-up response after tool call
                        follow_up_content = st.session_state.gemini_messages[-1]["content"]
                        updated_contents = contents + [
                            types.Content(
                                role="model", 
                                parts=[types.Part(function_call=function_call)]
                            ),
                            types.Content(
                                role="user", 
                                parts=[types.Part(text=f"Tool {function_call.name} returned: {follow_up_content}")]
                            )
                        ]
                        
                        follow_up_response = await asyncio.to_thread(
                            st.session_state.gemini_client.models.generate_content,
                            model="gemini-2.0-flash",
                            contents=updated_contents,
                            config=types.GenerateContentConfig(temperature=0.7),
                        )
                        
                        # Display and save follow-up response
                        response_text = follow_up_response.text
                        message_placeholder.markdown(response_text)
                        st.session_state.gemini_messages.append({"role": "model", "content": response_text})
                        
                    else:
                        # Display and save text response
                        response_text = response.text
                        if response_text:
                            message_placeholder.markdown(response_text)
                            st.session_state.gemini_messages.append({"role": "model", "content": response_text})
                        else:
                            error_msg = "Empty response from Gemini"
                            message_placeholder.markdown(error_msg)
                            st.session_state.gemini_messages.append({"role": "model", "content": error_msg})
                
                # Run the async call
                asyncio.run(call_gemini())
                
            except Exception as e:
                st.error(f"Error in Gemini chat: {str(e)}")
                st.session_state.gemini_messages.append({"role": "model", "content": f"Error: {str(e)}"})

if __name__ == "__main__":
    main() 