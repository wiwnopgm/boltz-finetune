#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Boltz-1 MCP Client

This client provides a Python interface to interact with the Boltz-1 MCP server.
It allows you to:
- Access Boltz-1 documentation and resources
- Prepare inputs for Boltz-1
- Run training and fine-tuning
- Perform inference
- Analyze results
"""

import os
import json
import logging
import asyncio
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import AsyncExitStack

from pydantic_ai import RunContext, Tool as PydanticTool
from pydantic_ai.tools import ToolDefinition
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class BoltzMCPClient:
    """Client for interacting with the Boltz-1 MCP server."""

    def __init__(self, name: str, config_path: Optional[str] = None) -> None:
        """Initialize the Boltz MCP client.
        
        Args:
            config_path: Optional path to configuration file. If not provided,
                        will look for boltz_config.json in the current directory.
        """
        self.name: str = name
        self.config_path = config_path or "boltz_config.json"
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools: List[PydanticTool] = []
        
        # Load environment variables
        load_dotenv()

    async def connect(self) -> List[PydanticTool]:
        """Connect to the Boltz MCP server and initialize tools.
        
        Returns:
            List of PydanticTool instances for interacting with the server.
        """
        try:
            # Load configuration
            with open(self.config_path, "r") as f:
                config = json.load(f)
            
            # Get server command and arguments
            command = shutil.which("python") or "python"
            server_script = Path(__file__).parent / "boltz_mcp_server.py"
            
            if not server_script.exists():
                raise FileNotFoundError(f"Server script not found at {server_script}")
            
            # Set up server parameters
            server_params = StdioServerParameters(
                command=command,
                args=[str(server_script)],
                env=os.environ.copy()
            )
            
            # Connect to server
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            
            # Initialize session
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await self.session.initialize()
            
            # Get available tools
            tools = (await self.session.list_tools()).tools
            self.tools = [self._create_tool_instance(tool) for tool in tools]
            
            return self.tools
            
        except Exception as e:
            logging.error(f"Failed to connect to Boltz MCP server: {e}")
            await self.cleanup()
            raise

    def _create_tool_instance(self, tool: MCPTool) -> PydanticTool:
        """Create a PydanticTool instance from an MCP tool.
        
        Args:
            tool: The MCP tool definition
            
        Returns:
            A PydanticTool instance for executing the tool
        """
        async def execute_tool(**kwargs: Any) -> Any:
            if not self.session:
                raise RuntimeError("Not connected to server")
            return await self.session.call_tool(tool.name, arguments=kwargs)

        async def prepare_tool(ctx: RunContext, tool_def: ToolDefinition) -> ToolDefinition | None:
            tool_def.parameters_json_schema = tool.inputSchema
            return tool_def
        
        return PydanticTool(
            execute_tool,
            name=tool.name,
            description=tool.description or "",
            takes_ctx=False,
            prepare=prepare_tool
        )

    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        if self.session:
            try:
                await self.exit_stack.aclose()
                self.session = None
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")

    async def __aenter__(self) -> "BoltzMCPClient":
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.cleanup()

# Example usage
async def main():
    """Example usage of the Boltz MCP client."""
    async with BoltzMCPClient(name="boltz-client") as client:
        # Get available tools
        tools = client.tools
        
        # Print available tools
        print("\nAvailable Boltz-1 tools:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
        
        # Example: Get documentation
        docs_tool = next(t for t in tools if t.name == "get_documentation")
        docs = await docs_tool.execute_tool()
        print("\nDocumentation:")
        print(docs)

if __name__ == "__main__":
    asyncio.run(main()) 