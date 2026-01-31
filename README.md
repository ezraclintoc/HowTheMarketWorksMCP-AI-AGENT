# HTMW AI Agent

An autonomous stock trading agent for HowTheMarketWorks (HTMW). 
It uses **Ollama** for decision-making and a custom **MCP (Model Context Protocol)** server for browser-based automation via Selenium.

## Features
- **Autonomous Trading**: Researches and executes trades based on a strategy (Momentum, Mean Reversion, Catalyst, Diversified).
- **Portfolio Management**: Monitors net worth, cash balance, and open positions.
- **Risk Control**: Configurable risk levels (Conservative, Moderate, Aggressive) with position sizing and cash buffer rules.
- **MCP Server**: Provides tools for ticker research, market movers, analyst ratings, and trade execution.
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
   Create a `.env` file with your credentials:
   ```env
   USERNAME="your_htmw_username"
   PASSWORD="your_htmw_password"
   TRADING_STRATEGY=MOMENTUM
   RISK_LEVEL=MODERATE
   COMMISSION_RATE=0
   ```

4. **Running the Agent**:
   ```bash
   # Autonomous Mode
   python ollama_trader.py --auto --colorize
   
   # Verbose Mode (see research)
   python ollama_trader.py --auto --verbose 1 --colorize
   ```

## Files
- `mcp_scrape.py`: The MCP server that handles Selenium automation.
- `ollama_trader.py`: The main loop that interacts with Ollama and the MCP server.
- `test_tools.py`: A script to verify tool functionality.
