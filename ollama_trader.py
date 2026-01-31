import asyncio
import os
import json
import argparse
import time
import sys
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama
from dotenv import load_dotenv

import random
import itertools

load_dotenv()

# Configuration
DEFAULT_MODEL = "gpt-oss:20b"
MCP_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_scrape.py")
VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv/bin/python")

# Global state for flags
config = {
    "verbose_level": 0,
    "silent": False,
    "debug": False,
    "colorize": False,
    "model": DEFAULT_MODEL,
    "strategy": os.getenv("TRADING_STRATEGY", "DIVERSIFIED").upper(),
    "risk_level": os.getenv("RISK_LEVEL", "MODERATE").upper(),
    "commission_rate": os.getenv("COMMISSION_RATE", "0")
}

STRATEGY_MAP = {
    "MOMENTUM": "Focus on high-volume tickers, top gainers, and stocks showing strong upward price action. Ride the trend.",
    "MEAN_REVERSION": "Look for oversold stocks, top losers, and stocks trading near 52-week lows that show signs of stabilizing. Bet on the bounce.",
    "CATALYST": "Prioritize tickers with high news volume, recent earnings reports, or major macro event impact (like Fed announcements).",
    "DIVERSIFIED": "Maintain a balanced approach across different sectors. Combine growth and value while avoiding over-concentration."
}

RISK_MAP = {
    "CONSERVATIVE": {
        "desc": "Prioritize capital preservation. Focus on large-cap ETFs and blue-chip stocks. Avoid high volatility.",
        "max_pos": 2000,
        "cash_buffer": "50%"
    },
    "MODERATE": {
        "desc": "Balanced growth and risk. Mix of ETFs and individual stocks with proven performance.",
        "max_pos": 5000,
        "cash_buffer": "20%"
    },
    "AGGRESSIVE": {
        "desc": "High conviction, high reward. Focus on volatile movers and aggressive position sizing for maximum growth.",
        "max_pos": 10000,
        "cash_buffer": "5%"
    }
}

class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    @classmethod
    def paint(cls, text, color):
        if not config["colorize"]:
            return text
        return f"{color}{text}{cls.END}"

TAGLINES = [
    "Analyzing market trends...",
    "Scanning for momentum setups...",
    "Researching ticker catalysts...",
    "Calculating optimal position sizes...",
    "Reviewing latest financial news...",
    "Evaluating risk parameters...",
    "Optimizing portfolio allocation...",
    "Checking technical indicators...",
    "Monitoring market movers...",
    "Simulating trade outcomes..."
]

class AISpinner:
    def __init__(self, message="Thinking"):
        self.message = message
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.stop_event = asyncio.Event()
        self.task = None

    async def spin(self):
        start_time = time.time()
        tagline = random.choice(TAGLINES)
        
        while not self.stop_event.is_set():
            # Update tagline every 3 seconds
            if time.time() - start_time > 3:
                tagline = random.choice(TAGLINES)
                start_time = time.time()
                
            spinner_char = Color.paint(next(self.spinner), Color.CYAN)
            # Use \033[K to clear from cursor to end of line to prevent ghosting
            sys.stdout.write(f"\r{spinner_char} {tagline}\033[K")
            sys.stdout.flush()
            await asyncio.sleep(0.1)
        # Final clear of the line
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()

    async def __aenter__(self):
        if config["verbose_level"] == 0 and not config["silent"] and not config["debug"]:
            self.task = asyncio.create_task(self.spin())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.task:
            self.stop_event.set()
            await self.task

def log(msg, level="info"):
    """
    Custom logger that respects silent/verbose/debug flags.
    Levels: 
    - info: Always show (unless silent)
    - verbose1: Show when level >= 1
    - verbose2: Show when level >= 2
    - debug: Show when debug is True
    - error: Always show to stderr
    """
    if config["silent"]:
        if level == "error":
            print(f"ERROR: {msg}", file=sys.stderr)
        return

    if level == "debug" and not config["debug"]:
        return
    
    if level == "info":
        print(msg)
    elif level == "verbose1" and (config["verbose_level"] >= 1 or config["debug"]):
        # Colorize tool calling marker
        if msg.startswith("[*]"):
            msg = Color.paint("[*]", Color.BLUE) + msg[3:]
        print(msg)
    elif level == "verbose2" and (config["verbose_level"] >= 2 or config["debug"]):
        if msg.startswith("[+]"):
            msg = Color.paint("[+]", Color.GREEN) + msg[3:]
        print(msg)
    elif level == "error":
        print(f"{Color.paint('ERROR:', Color.RED)} {msg}", file=sys.stderr)

async def process_ai_response(session, messages, ollama_tools):
    """Handles the LLM chat and tool execution logic."""
    log(f"Calling Ollama with model {config['model']}...", level="debug")
    
    # Use a single spinner for the entire process if in low-verbosity mode
    # to avoid flickering when switching between thinking and tool execution.
    spinner = AISpinner()
    
    try:
        if config["verbose_level"] == 0 and not config["silent"] and not config["debug"]:
            await spinner.__aenter__()

        # 1. Initial LLM call
        response = await asyncio.to_thread(
            ollama.chat,
            model=config["model"],
            messages=messages,
            tools=ollama_tools,
        )

        # Process potential tool calls
        while response.get('message', {}).get('tool_calls'):
            # If AI has content, print it (this will break the spinner line, which is fine/intended)
            if response['message'].get('content'):
                print(f"\nAI: {response['message']['content']}\n")
                response['message']['content'] = ""

            messages.append(response['message'])
            
            for tool_call in response['message']['tool_calls']:
                tool_name = tool_call['function']['name']
                arguments = tool_call['function']['arguments']
                
                # IMPORTANT: Always show trade executions even in verbose 0
                if tool_name == "trade_stock":
                    log(f"[*] Executing Trade: {arguments}", level="info")
                else:
                    log(f"[*] Calling tool: {tool_name}({arguments})", level="verbose1")
                
                try:
                    tool_result = await session.call_tool(tool_name, arguments)
                    content = str(tool_result.content)
                    
                    if tool_name == "trade_stock":
                        log(f"[+] Trade Status: {content}", level="info")
                    else:
                        log(f"[+] Tool Raw Result: {content}", level="verbose2")
                except Exception as e:
                    content = f"Error executing tool: {str(e)}"
                    log(f"Tool execution failed: {e}", level="error")
                
                messages.append({
                    'role': 'tool',
                    'content': content,
                })
            
            # Get next response from LLM
            response = await asyncio.to_thread(
                ollama.chat,
                model=config["model"],
                messages=messages,
                tools=ollama_tools,
            )

        # Print final assistant message if any remaining
        final_message = response['message']['content']
        if final_message:
            print(f"\nAI: {final_message}\n")
        
        messages.append(response['message'])
        return messages

    finally:
        if spinner.task:
            await spinner.__aexit__(None, None, None)

async def run_trader(auto_mode=False):
    # 1. Connect to the MCP Server
    server_params = StdioServerParameters(
        command=VENV_PYTHON,
        args=[MCP_SERVER_PATH],
        env=os.environ.copy()
    )

    async with AsyncExitStack() as stack:
        log(f"Connecting to MCP server at {MCP_SERVER_PATH}...", level="verbose")
        results = await stack.enter_async_context(stdio_client(server_params))
        read, write = results
        session = await stack.enter_async_context(ClientSession(read, write))
        
        # Initialize the session
        await session.initialize()
        
        # 2. Get available tools from MCP
        tools_response = await session.list_tools()
        mcp_tools = tools_response.tools
        
        # Convert MCP tools to Ollama tool format
        ollama_tools = []
        for tool in mcp_tools:
            ollama_tools.append({
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.inputSchema
                }
            })
        
        log(f"Connected to MCP Server. Found {len(ollama_tools)} tools.", level="info")
        if config["debug"]:
            log(f"Tools available: {[t.name for t in mcp_tools]}", level="debug")
        
        messages = [
            {"role": "system", "content": "You are a professional stock trading assistant. You have access to the HowTheMarketWorks trading platform through various tools. Use them to help the user manage their portfolio, check news, and execute trades."}
        ]

        if auto_mode:
            banner = "\n--- HTMW AI Trader (AUTO MODE) Started ---"
            log(Color.paint(banner, Color.BOLD + Color.PURPLE), level="info")
            log("Press Ctrl+C to stop.\n", level="info")
            
            cycle_count = 1
            while True:
                cycle_header = f"=== Starting Trade Cycle #{cycle_count} ==="
                log(Color.paint(cycle_header, Color.BOLD + Color.YELLOW), level="info")
                
                strategy_desc = STRATEGY_MAP.get(config["strategy"], STRATEGY_MAP["DIVERSIFIED"])
                risk_data = RISK_MAP.get(config["risk_level"], RISK_MAP["MODERATE"])
                
                auto_prompt = (
                    f"You are the Lead Portfolio Manager for this automated trading system. Your objective is build capital and manage risk autonomously.\n"
                    f"SYSTEM SETTINGS: Strategy={config['strategy']} | Risk={config['risk_level']} | Commission=${config['commission_rate']} per trade\n\n"
                    "### OPERATIONAL MISSION:\n"
                    f"1. **RESEARCH**: Use all available tools to analyze the portfolio, identify market movers, and audit specific tickers. Deep-dive into technicals and news.\n"
                    f"2. **STRATEGY ALIGNMENT**: Verify that potential trades align with the '{config['strategy']}' strategy: {strategy_desc}\n"
                    f"3. **EXECUTE**: Immediately execute the necessary `trade_stock` calls based on your research. Do not wait or ask for confirmation; your role is to manage this portfolio independently.\n"
                    f"4. **STATUS**: Provide a concise summary of your actions at the end of the cycle.\n\n"
                    "### FIDUCIARY RULES:\n"
                    "1. **Autonomy**: You are in sole control. Do not provide advice or suggestions to the user; instead, execute the actions required by your strategy.\n"
                    f"2. **Position Sizing**: Limit total exposure to ${risk_data['max_pos']} per ticker. Scale individual trades based on conviction.\n"
                    f"3. **Risk Management**: Maintain a protective cash buffer of {risk_data['cash_buffer']} for account stability.\n"
                    "4. **Trade Actions**: Use only 'Buy' and 'Sell' orders.\n"
                    "5. **Performance Bias**: Proactively deploy capital when your strategy identifies high-probability setups. Avoid unnecessary idleness in cash."
                )
                
                messages.append({"role": "user", "content": auto_prompt})
                messages = await process_ai_response(session, messages, ollama_tools)
                
                # Keep history manageable - keep system prompt + last 10 messages
                if len(messages) > 15:
                    messages = [messages[0]] + messages[-14:]
                
                log(f"=== Cycle #{cycle_count} complete. Waiting 15 minutes... ===\n", level="info")
                cycle_count += 1
                await asyncio.sleep(900) # Wait 15 minutes between cycles
        else:
            log("\n--- HTMW AI Trader (Ollama + MCP) Ready ---", level="info")
            log("Type 'exit' to quit.\n", level="info")

            while True:
                try:
                    user_input = input("You: ")
                except EOFError:
                    break
                    
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                messages.append({"role": "user", "content": user_input})
                messages = await process_ai_response(session, messages, ollama_tools)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTMW AI Trader with Ollama")
    parser.add_argument("--auto", action="store_true", help="Run in automatic trading mode")
    parser.add_argument("--verbose", type=int, nargs="?", const=1, default=0, help="Verbose level: 0=regular, 1=tool calls, 2=tool results")
    parser.add_argument("--silent", action="store_true", help="Suppress all output except errors")
    parser.add_argument("--debug", action="store_true", help="Show internal debug messages and tool traces")
    parser.add_argument("--colorize", action="store_true", help="Enable colorized output")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    # Update global config
    config["verbose_level"] = args.verbose
    config["silent"] = args.silent
    config["debug"] = args.debug
    config["colorize"] = args.colorize
    config["model"] = args.model

    try:
        asyncio.run(run_trader(auto_mode=args.auto))
    except KeyboardInterrupt:
        log("\nTrader stopped by user.", level="info")
    except Exception as e:
        log(f"Critical error: {e}", level="error")
