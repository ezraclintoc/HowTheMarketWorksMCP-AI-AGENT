# HTMW AI Agent

An autonomous stock trading agent for **HowTheMarketWorks (HTMW)**. 
Built with a modular, 5-phase pipeline using **Ollama** for decision-making and a custom **MCP (Model Context Protocol)** server for browser automation via Selenium.

## 🚀 Features

- **5-Phase Trading Pipeline**: Structured architecture (Intelligence → Analysis → Strategy → Risk Gate → Execution) ensures robust decision-making.
- **Persistence Layer**: `trading_journal.json` and `watchlist.json` allow the agent to learn from past trades and monitor symbols over time.
- **Deterministic Intelligence**: Critical data (portfolio, positions, movers) is gathered via code, not LLM guesswork.
- **Robust Tool Calling**: Supports both native API tool calls and fallback text-parsing for LLMs that output raw JSON.
- **Risk Control**: Deterministic risk gate in Python code (not LLM) enforces position sizing and cash buffers.
- **Sleek TUI**: `config_tui.py` provides an easy way to manage credentials, model providers, and strategy.

## 🛠️ Setup

1. **Prerequisites**:
   - Python 3.10+
   - Chrome Browser installed.
   - Ollama (or an OpenAI-compatible provider like vLLM, llama.cpp).

2. **Installation**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configuration**:
   Run the TUI to set up your environment:
   ```bash
   python config_tui.py
   ```
   Alternatively, copy `.env.example` to `.env` and fill in your details.

4. **Running**:
   ```bash
   python ollama_trader.py --auto --colorize --verbose 2
   ```

## 🐳 Docker & MCP Integration

### Running as an MCP Server in Docker

To run the `mcp_scrape.py` as an MCP server within a Docker container:

1. **Build the Image**:
   ```bash
   docker build -t htmw-mcp-server .
   ```

2. **Run the Container**:
   ```bash
   docker run -d --name htmw-mcp \
     -e HTMW_USERNAME=your_username \
     -e HTMW_PASSWORD=your_password \
     -p 8000:8000 \
     htmw-mcp-server
   ```

### Connecting External Clients (Claude/Cursor)

To connect this agent to Claude Desktop or Cursor as an MCP server:

Add the following to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "htmw-trader": {
      "command": "python",
      "args": ["/path/to/mcp_scrape.py"],
      "env": {
        "HTMW_USERNAME": "your_user",
        "HTMW_PASSWORD": "your_password"
      }
    }
  }
}
```

## 📐 Architecture: 5-Phase Pipeline

1. **Phase 1: Intelligence** — Code-driven gathering of portfolio and market movers.
2. **Phase 2: Analysis** — LLM research on candidate symbols (news, technicals, analyst sentiment).
3. **Phase 3: Strategy** — LLM determines trades based on analysis and historical journal context.
4. **Phase 4: Risk Gate** — Python code enforces safety rules (position limits, cash buffer).
5. **Phase 5: Execution** — Deterministic buy/sell actions and journaling.

## 📂 Files
- `mcp_scrape.py`: MCP server implementation (Selenium).
- `ollama_trader.py`: The trading engine and 5-phase pipeline.
- `AGENTS.md`: Deep-dive documentation on agent roles and tools.
- `trading_journal.json`: Persistent history of trading cycles.
- `watchlist.json`: Persistent list of monitored tickers.
