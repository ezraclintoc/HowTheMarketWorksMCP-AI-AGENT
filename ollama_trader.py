import json
import asyncio
import argparse
import sys
import threading
import ollama
from openai import AsyncOpenAI
import trading_engine
from trading_engine import run_trader, config, log, Color

class OllamaProvider:
    def __init__(self, model):
        self.client = ollama.AsyncClient()
        self.model = model

    async def chat(self, messages, tools=None, model=None):
        return await self.client.chat(model=model or self.model, messages=messages, tools=tools)

class OpenAIProvider:
    def __init__(self, base_url, api_key, model):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    async def chat(self, messages, tools=None, model=None):
        response_raw = await self.client.chat.completions.create(model=model or self.model, messages=messages, tools=tools)
        msg = response_raw.choices[0].message
        formatted = {'message': {'role': 'assistant', 'content': msg.content or "", 'tool_calls': []}}
        if msg.tool_calls:
            for tc in msg.tool_calls:
                formatted['message']['tool_calls'].append({'function': {'name': tc.function.name, 'arguments': json.loads(tc.function.arguments)}})
        return formatted

def get_provider():
    if config["provider"] == "ollama" and "localhost" in config["base_url"] and config["base_url"].endswith("11434/v1"):
        return OllamaProvider(config["model"])
    else:
        return OpenAIProvider(config["base_url"], config["api_key"], config["model"])

def main():
    parser = argparse.ArgumentParser(description="HTMW AI Trader - Professional Workspace")
    parser.add_argument("--auto", action="store_true", help="Run in automatic trading mode (Headless CLI)")
    parser.add_argument("--gui", action="store_true", help="Launch the Professional Desktop GUI (PySide6)")
    parser.add_argument("--cli", action="store_true", help="Launch the Interactive Analyst CLI (Rich)")
    parser.add_argument("--mcp-only", action="store_true", help="Run only the MCP server (for external clients)")
    
    parser.add_argument("--verbose", type=int, nargs="?", const=1, default=0, help="Verbose level: 0=regular, 1=tool calls, 2=tool results")
    parser.add_argument("--silent", action="store_true", help="Suppress all output except errors")
    parser.add_argument("--debug", action="store_true", help="Show internal debug messages")
    parser.add_argument("--colorize", action="store_true", default=True, help="Enable colorized output")
    parser.add_argument("--model", type=str, default=trading_engine.DEFAULT_MODEL, help=f"LLM model to use")
    args = parser.parse_args()

    # Handle MCP-only mode immediately
    if args.mcp_only:
        print("Starting HTMW MCP Server in standalone mode...")
        print(f"Server location: {trading_engine.MCP_SERVER_PATH}")
        print("You can now connect to this server via standard MCP clients (Studio, Claude Desktop, etc.)")
        # Run the scraper as a main process
        import subprocess
        try:
            subprocess.run([trading_engine.VENV_PYTHON, trading_engine.MCP_SERVER_PATH])
        except KeyboardInterrupt:
            print("\nMCP Server stopped.")
        return

    config.update({
        "verbose_level": args.verbose,
        "silent": args.silent,
        "debug": args.debug,
        "colorize": args.colorize,
        "model": args.model
    })

    if args.gui:
        # Launch the PySide6 GUI
        # Note: GUI runs the trader in its own thread internally
        import trader_gui
        from PySide6.QtWidgets import QApplication
        app = QApplication(sys.argv)
        provider = get_provider()
        window = trader_gui.TradingGUI(provider=provider)
        window.show()
        sys.exit(app.exec())
        
    elif args.cli:
        # Future enhancement: A more interactive Rich CLI
        # For now, we reuse the existing auto_mode loop with better formatting
        print(Color.paint("HTMW Trading Station - Analyst CLI", Color.BOLD + Color.CYAN))
        try:
            asyncio.run(run_trader(auto_mode=True, provider=get_provider()))
        except KeyboardInterrupt:
            print("\nTrader stopped by user.")
        except Exception as e:
            log(f"Critical error: {e}", level="error")
            
    elif args.auto:
        # Pure CLI auto-drive
        try:
            asyncio.run(run_trader(auto_mode=True, provider=get_provider()))
        except KeyboardInterrupt:
            log("\nTrader stopped by user.", level="info")
        except Exception as e:
            log(f"Critical error: {e}", level="error")
    else:
        # Interactive Helper Mode
        print(Color.paint("HTMW Trading Station - Interactive Helper", Color.BOLD + Color.PURPLE))
        try:
            asyncio.run(run_trader(auto_mode=False, provider=get_provider()))
        except KeyboardInterrupt:
            log("\nAssistant stopped by user.", level="info")
        except Exception as e:
            log(f"Critical error: {e}", level="error")

if __name__ == "__main__":
    main()
