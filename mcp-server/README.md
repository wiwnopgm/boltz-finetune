# BioAI Agent on Model Context Protocal (MCP Server)

This repository contains a Streamlit application for interacting with the Boltz-1 protein structure prediction model. The application provides a user-friendly web interface for using Boltz-1 functions.

## Features

- **Interactive Chat Interface**: Chat with an AI assistant that can help you work with Boltz-1
- **Documentation Access**: Get information about Boltz-1 and available tools
- **Input Preparation**: Process structure files (PDB, CIF) and sequence files (FASTA)
- **Model Fine-tuning**: Fine-tune Boltz-1 models using LoRA or full fine-tuning
- **Inference**: Run inference with Boltz-1 models
- **Results Analysis**: Analyze results from Boltz-1 predictions
- **MCP Server Integration**: Access the Boltz-1 MCP server for advanced functionality
- **Gemini-powered BioAI Agent**: Interact with a Gemini LLM that has access to Boltz-1 tools

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd mcp-server-demo
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Note: If you encounter any issues with the pydantic-ai package, you can try installing a specific version:
   ```bash
   pip install pydantic-ai==0.0.55
   ```

3. Create a `.env` file with your API keys:
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys
   ```

   The `.env` file should contain:
   ```
   LLM_API_KEY=your-openai-api-key-here
   MODEL_CHOICE=gpt-4o-mini
   BASE_URL=https://api.openai.com/v1
   GEMINI_API_KEY=your-gemini-api-key-here
   ```

## Usage

### Streamlit Application

1. Start the Streamlit application using the provided script:

   ```bash
   streamlit run boltz_streamlit.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Start chatting with the Boltz-1 AI assistant!

### Gemini BioAI Agent

The repository includes a Gemini-powered BioAI agent that can interact with the Boltz-1 MCP server and execute tools on your behalf.

1. Make sure you have set your GEMINI_API_KEY in the .env file

2. Run the Gemini MCP client:
   ```bash
   cd mcp-server
   python gemini_mcp_client.py
   ```

3. Wait for the initialization to complete. You'll see a welcome message from the assistant.

4. Chat with the BioAI agent by typing your queries. For example:
   - "What tools do you have access to?"
   - "How can I prepare protein structure data for Boltz-1?"
   - "How do I run inference with a Boltz-1 model?"
   - "Can you help me analyze my prediction results?"

5. To exit the chat, type "exit", "quit", "bye", or "goodbye".

## MCP Server

The Boltz-1 MCP (Model Control Protocol) Server provides a comprehensive set of tools and resources for working with the Boltz-1 architecture. It enables programmatic access to Boltz-1 functionality through a standardized interface.

### Running the MCP Server

The MCP server can be started manually:

```bash
cd mcp-server
python mcp_server.py
```

The server runs locally and provides an API that clients can connect to.

### MCP Server Features

- **Resource Access**: Access documentation, configuration templates, and model information
- **Remote Execution**: Run Boltz-1 commands on remote servers via SSH
- **Data Processing**: Process training data for Boltz-1 architecture
- **Training Pipeline**: Run training on Boltz-1 models with custom configurations
- **Redis Server Management**: Start and manage Redis servers for CCD and Taxonomy data

### Available MCP Resources

- `boltz://docs` - Access documentation about Boltz-1 and the server
- `boltz://config/{config_name}` - Access configuration templates (train_full, finetune_lora, inference)
- `boltz://model/{model_path}` - Access information about a model checkpoint

### Available MCP Tools

- `process_train_data` - Prepare inputs for Boltz-1 architecture
- `run_training` - Run training on Boltz-1 model
- `run_inference` - Run inference with Boltz-1 model
- `get_prediction_results` - Analyze results from Boltz-1 predictions
- `connect_ssh` - Configure SSH access to a remote cluster
- `command_remote_dir` - Run commands in a specific directory on a remote server
- `interpret_command` - Get terminal commands from natural language descriptions

### MCP Prompts

The MCP server also provides predefined prompts for common tasks:

- `train` - Guide for running the Boltz-1 training pipeline
- `predict` - Guide for running the Boltz-1 inference pipeline
- `analyze_data` - Guide for analyzing data from Boltz-1 predictions

## Example Queries for BioAI Agents

Here are some example queries you can use with either the Streamlit-based assistant or the Gemini-powered BioAI agent:

- "What is Boltz-1 and what can it do?"
- "Show me the documentation for Boltz-1"
- "How do I prepare inputs for Boltz-1?"
- "How do I fine-tune a Boltz-1 model?"
- "How do I run inference with Boltz-1?"
- "How do I analyze results from Boltz-1 predictions?"
- "How do I use the MCP server to process training data?"
- "How do I start the Redis servers for Boltz-1?"
- "Can you explain the main parameters in the training configuration?"

Future Ideas

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 