#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Boltz MCP Client with Gemini Chat Interface
Provides a chat interface to interact with Google's Gemini model and Boltz MCP server tools.
"""

import os
import sys
import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Setup paths and configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
SERVER_PATH = SCRIPT_DIR / "mcp_server.py"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

DEFAULT_MODEL = "gemini-2.0-flash-lite"

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable is not set.")
    sys.exit(1)
    
# Initialize clients
client = genai.Client(api_key=GEMINI_API_KEY)
server_params = StdioServerParameters(
    command="python",
    args=[str(SERVER_PATH)],
    env={},
)

def convert_mcp_to_gemini_schema(mcp_schema: dict, tool_name: str, tool_description: str) -> dict:
    """
    Convert MCP schema format to Gemini-friendly tool schema format.
    
    Args:
        mcp_schema: Raw MCP schema dictionary
        tool_name: Name of the tool
        tool_description: Description of the tool
        
    Returns:
        Dictionary in Gemini-friendly schema format
    """
    gemini_schema = {
        "name": tool_name,
        "description": tool_description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
    
    # Process properties
    for prop_name, prop_details in mcp_schema["properties"].items():
        # Check if property is nullable (has anyOf with null)
        is_nullable = False
        if "anyOf" in prop_details:
            is_nullable = any(t.get("type") == "null" for t in prop_details["anyOf"])
            # Get the non-null type
            prop_type = next(t["type"] for t in prop_details["anyOf"] if t["type"] != "null")
        else:
            prop_type = prop_details["type"]
            
        # Add to properties
        gemini_schema["parameters"]["properties"][prop_name] = {
            "type": prop_type,
            "description": prop_details.get("title", prop_name)
        }
        
        # Add to required list if not nullable and in original required list
        if not is_nullable and prop_name in mcp_schema.get("required", []):
            gemini_schema["parameters"]["required"].append(prop_name)
    
    return gemini_schema

async def process_function_call(session, function_call, user_input, contents, chat_history, live):
    """Process a function call from Gemini and handle the response."""
    print(f"\nFunction to call: {function_call.name}")
    print(f"Arguments: {function_call.args}")
    
    try:
        # Process tool arguments
        tool_args = function_call.args
        print(f"Tool arguments: {tool_args}")
        
        # Call the tool with arguments as a single parameter
        result = await session.call_tool(function_call.name, tool_args)
        print(f"\nTool result: {result}")
        
        # Update chat history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "model", "content": "I'll help you with that."})
        chat_history.append({"role": "user", "content": f"Tool {function_call.name} returned: {result}"})
        
        # Get follow-up response
        follow_up_contents = [
            *contents,
            types.Content(role="model", parts=[types.Part(text="I'll help you with that.")]),
            types.Content(role="user", parts=[types.Part(text=f"Tool {function_call.name} returned: {result}")]),
        ]
        
        follow_up_response = await asyncio.to_thread(
            client.models.generate_content,
            model=DEFAULT_MODEL,
            contents=follow_up_contents,
            config=types.GenerateContentConfig(temperature=0.7),
        )
        
        # Display response
        if follow_up_response and follow_up_response.text:
            live.update(Markdown(follow_up_response.text))
            chat_history.append({"role": "model", "content": follow_up_response.text})
        else:
            live.update(Markdown("I've completed the requested operation."))
            chat_history.append({"role": "model", "content": "I've completed the requested operation."})
            
    except Exception as e:
        error_msg = f"Error executing tool: {str(e)}"
        print(f"\n{error_msg}")
        live.update(Markdown(error_msg))
        chat_history.append({"role": "model", "content": error_msg})

async def run_chat():
    """Run a chat session with Gemini and MCP tools."""
    print("=== Boltz MCP Client with Gemini Chat ===")
    print("Type 'exit' to quit the chat")
    
    console = Console()
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()
            
            # Get and process tools
            mcp_tools = await session.list_tools()
            function_declarations = [convert_mcp_to_gemini_schema(tool.inputSchema, tool.name, tool.description) for tool in mcp_tools.tools]
            
            for i, tool in enumerate(mcp_tools.tools):
                if tool.name == "run_training":
                    print(function_declarations[i])
            tools = types.Tool(function_declarations=function_declarations)
            
            # Display welcome message
            print("\n[Assistant]")
            with Live('', console=console, vertical_overflow='visible') as live:
                welcome_message = (
                    "I'm an assistant powered by Gemini that has access to tools for working with the Boltz-1 architecture. "
                    "I can help you with data processing, model training, fine-tuning, running inference, and analyzing results. "
                    "How can I assist you today?"
                )
                live.update(Markdown(welcome_message))
            
            # Chat history
            chat_history = []
            
            while True:
                # Get user input
                user_input = input("\n[You] ")
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print("Goodbye!")
                    break
                
                try:
                    print("\n[Assistant]")
                    
                    # Format conversation history
                    contents = []
                    for msg in chat_history:
                        role = "user" if msg["role"] == "user" else "model"
                        contents.append(types.Content(
                            role=role,
                            parts=[types.Part(text=msg["content"])]
                        ))
                    
                    # Add current user input
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part(text=user_input)]
                    ))
                    
                    # Send request to Gemini
                    with Live('', console=console, vertical_overflow='visible') as live:
                        response = await asyncio.to_thread(
                            client.models.generate_content,
                            model="gemini-2.0-flash",
                            contents=contents,
                            config=types.GenerateContentConfig(
                                temperature=0.7,
                                tools=[tools],
                            ),
                        )
                        
                        # Validate response
                        if not response or not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
                            raise ValueError("Invalid response from Gemini API")
                        
                        # Check for function call
                        if hasattr(response.candidates[0].content.parts[0], 'function_call') and response.candidates[0].content.parts[0].function_call:
                            await process_function_call(
                                session, 
                                response.candidates[0].content.parts[0].function_call,
                                user_input, 
                                contents, 
                                chat_history, 
                                live
                            )
                        else:
                            # Handle text response
                            response_text = response.text
                            if response_text is None:
                                raise ValueError("Response text is None")
                                
                            live.update(Markdown(response_text))
                            chat_history.append({"role": "user", "content": user_input})
                            chat_history.append({"role": "model", "content": response_text})
                
                except Exception as e:
                    print(f"\n[Error] An error occurred: {str(e)}")

async def main():
    """Main function to run the client with chat interface."""
    try:
        await run_chat()
    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting due to user interrupt...")
    except Exception as e:
        print(f"Fatal error: {str(e)}") 