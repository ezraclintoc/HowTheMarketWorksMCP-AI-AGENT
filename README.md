# HTMW AI Agent

An autonomous stock trading agent for HowTheMarketWorks (HTMW). 
It uses **Ollama** for decision-making and a custom **MCP (Model Context Protocol)** server for browser-based automation via Selenium.

## Features
- **Multi-Agent "Trading Council"**: Orchestrates trades through specialized agents (Lead PM, Technical Analyst, Fundamental Analyst, Risk Officer).
- **Interactive TUI**: Sleek configuration manager for credentials, models, and strategy settings.
- **Multi-Provider Support**: Compatible with Ollama, llama.cpp, vLLM, and other OpenAI-compatible APIs.
- **Risk Control**: Built-in Risk Officer to enforce position sizing and cash buffers.
- **Dynamic Scraping**: MCP-integrated web scraping for real-time market data.
- **Interactive/Auto Mode**: Can run in full auto mode with a research loop or be used interactively.

## Setup

1. **Prerequisites**:
   - Python 3.10+
   - Ollama installed and running.
   - Chrome Browser.

2. **Installation**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configuration**:
   Use the new interactive TUI to set up your environment:
   ```bash
   python config_tui.py
   ```
   *Follow the prompts to set your HTMW credentials, model provider (Ollama/llama.cpp/vLLM), and trading strategy.*

4. **Running the Agent**:
   ```bash
   # Launch directly from the TUI or use:
   python ollama_trader.py --auto --colorize
   ```

## ⚙️ Key Concepts

### Multi-Provider Support
The agent now supports any OpenAI-compatible API. This includes **llama.cpp**, **vLLM**, and **LM Studio**. You can configure the `Base URL` and `API Key` via the TUI.

### Agentic Mode
- **Native (Multi-Agent Council)**: The Lead PM coordinates research with Technical and Fundamental analysts before a Risk Officer audits the final trade.
- **Classic (Single-Agent)**: A single autonomous agent makes decisions directly.

## Files
- `mcp_scrape.py`: The MCP server that handles Selenium automation.
- `ollama_trader.py`: The main loop that interacts with Ollama and the MCP server.
- `test_tools.py`: A script to verify tool functionality.
