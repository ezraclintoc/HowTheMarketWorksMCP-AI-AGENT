import asyncio
import os
import json
import argparse
import time
import sys
import re
from datetime import datetime
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama
from openai import AsyncOpenAI
from dotenv import load_dotenv

import random
import itertools

load_dotenv()

# Configuration
DEFAULT_MODEL = "gpt-oss:20b"
MCP_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_scrape.py")
VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv/bin/python")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
JOURNAL_PATH = os.path.join(PROJECT_DIR, "trading_journal.json")
WATCHLIST_PATH = os.path.join(PROJECT_DIR, "watchlist.json")

# Global state for flags
config = {
    "verbose_level": 0,
    "silent": False,
    "debug": False,
    "colorize": False,
    "provider": os.getenv("LLM_PROVIDER", "ollama").lower(),
    "base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    "api_key": os.getenv("LLM_API_KEY", "ollama"),
    "agent_model": os.getenv("AGENT_MODEL", os.getenv("LLM_MODEL", DEFAULT_MODEL)),
    "agentic_mode": os.getenv("AGENTIC_MODE", "NATIVE").upper(),
    "strategy": os.getenv("TRADING_STRATEGY", "DIVERSIFIED").upper(),
    "risk_level": os.getenv("RISK_LEVEL", "MODERATE").upper(),
    "commission_rate": float(os.getenv("COMMISSION_RATE", "0"))
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
        "cash_buffer": 0.50
    },
    "MODERATE": {
        "desc": "Balanced growth and risk. Mix of ETFs and individual stocks with proven performance.",
        "max_pos": 5000,
        "cash_buffer": 0.20
    },
    "AGGRESSIVE": {
        "desc": "High conviction, high reward. Focus on volatile movers and aggressive position sizing for maximum growth.",
        "max_pos": 10000,
        "cash_buffer": 0.05
    }
}

# ─────────────────────────────────────────────────────────────────────
# Display Utilities
# ─────────────────────────────────────────────────────────────────────

class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
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
            if time.time() - start_time > 3:
                tagline = random.choice(TAGLINES)
                start_time = time.time()
                
            spinner_char = Color.paint(next(self.spinner), Color.CYAN)
            sys.stdout.write(f"\r{spinner_char} {tagline}\033[K")
            sys.stdout.flush()
            await asyncio.sleep(0.1)
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
    if config["silent"]:
        if level == "error":
            print(f"ERROR: {msg}", file=sys.stderr)
        return

    if level == "debug" and not config["debug"]:
        return
    
    if level == "info":
        print(msg, flush=True)
    elif level == "verbose1" and (config["verbose_level"] >= 1 or config["debug"]):
        if msg.startswith("[*]"):
            msg = Color.paint("[*]", Color.BLUE) + msg[3:]
        print(msg, flush=True)
    elif level == "verbose2" and (config["verbose_level"] >= 2 or config["debug"]):
        if msg.startswith("[+]"):
            msg = Color.paint("[+]", Color.GREEN) + msg[3:]
        print(msg, flush=True)
    elif level == "error":
        print(f"{Color.paint('ERROR:', Color.RED)} {msg}", file=sys.stderr, flush=True)
    elif level == "phase":
        # Special formatting for phase headers
        print(f"\n{Color.paint('───', Color.DIM)} {Color.paint(msg, Color.BOLD + Color.PURPLE)} {Color.paint('───', Color.DIM)}", flush=True)

# ─────────────────────────────────────────────────────────────────────
# JSON Parsing Helpers
# ─────────────────────────────────────────────────────────────────────

def extract_json(text):
    """Defensively extract JSON from LLM output. Returns parsed object or None."""
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # 1. Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 2. Try extracting from markdown code blocks
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # 3. Try finding first {...} or [...]
    # Find JSON object
    brace_depth = 0
    start = -1
    for i, c in enumerate(text):
        if c == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif c == '}':
            brace_depth -= 1
            if brace_depth == 0 and start != -1:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    start = -1
    
    # Find JSON array
    bracket_depth = 0
    start = -1
    for i, c in enumerate(text):
        if c == '[':
            if bracket_depth == 0:
                start = i
            bracket_depth += 1
        elif c == ']':
            bracket_depth -= 1
            if bracket_depth == 0 and start != -1:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    start = -1
    
    return None

# ─────────────────────────────────────────────────────────────────────
# Persistence Layer
# ─────────────────────────────────────────────────────────────────────

def load_journal(limit=5):
    """Load the last N entries from the trading journal."""
    if not os.path.exists(JOURNAL_PATH):
        return []
    try:
        with open(JOURNAL_PATH, 'r') as f:
            entries = json.load(f)
        return entries[-limit:]
    except (json.JSONDecodeError, IOError):
        return []

def save_journal_entry(entry):
    """Append an entry to the trading journal."""
    entries = []
    if os.path.exists(JOURNAL_PATH):
        try:
            with open(JOURNAL_PATH, 'r') as f:
                entries = json.load(f)
        except (json.JSONDecodeError, IOError):
            entries = []
    
    entries.append(entry)
    
    # Keep only last 50 entries to prevent unbounded growth
    entries = entries[-50:]
    
    with open(JOURNAL_PATH, 'w') as f:
        json.dump(entries, f, indent=2)

def load_watchlist():
    """Load the persistent watchlist."""
    if not os.path.exists(WATCHLIST_PATH):
        return []
    try:
        with open(WATCHLIST_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def save_watchlist(symbols):
    """Save the watchlist."""
    # Deduplicate and limit
    symbols = list(dict.fromkeys(symbols))[:20]
    with open(WATCHLIST_PATH, 'w') as f:
        json.dump(symbols, f, indent=2)

# ─────────────────────────────────────────────────────────────────────
# LLM Interface
# ─────────────────────────────────────────────────────────────────────

async def llm_chat(messages, tools=None, model=None):
    """Unified chat helper for different providers."""
    model = model or config["model"]
    
    if config["provider"] == "ollama" and "localhost" in config["base_url"] and config["base_url"].endswith("11434/v1"):
        async_client = ollama.AsyncClient()
        response = await async_client.chat(
            model=model,
            messages=messages,
            tools=tools,
        )
        return response
    else:
        client = AsyncOpenAI(base_url=config["base_url"], api_key=config["api_key"])
        
        openai_tools = None
        if tools:
            openai_tools = tools
            
        response_raw = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
        )
        
        msg = response_raw.choices[0].message
        formatted_response = {
            'message': {
                'role': 'assistant',
                'content': msg.content or "",
                'tool_calls': []
            }
        }
        
        if msg.tool_calls:
            for tc in msg.tool_calls:
                formatted_response['message']['tool_calls'].append({
                    'function': {
                        'name': tc.function.name,
                        'arguments': json.loads(tc.function.arguments)
                    }
                })
        
        return formatted_response

async def process_ai_response(session, messages, ollama_tools, role_name="AI", model=None):
    """Handles the LLM chat and tool execution logic for a specific agent role."""
    current_model = model or config["model"]
    log(f"[{role_name}] Calling {config['provider']} with model {current_model}...", level="debug")
    
    spinner = AISpinner()
    
    try:
        if config["verbose_level"] == 0 and not config["silent"] and not config["debug"]:
            await spinner.__aenter__()

        response = await llm_chat(messages, tools=ollama_tools, model=current_model)

        while response.get('message', {}).get('tool_calls') or (response['message'].get('content') and '{"name":' in response['message'].get('content')):
            # FALLBACK: Detect tool calls output as text by the model
            raw_tool_calls = response['message'].get('tool_calls') or []
            content_text = response['message'].get('content', "")
            
            # If no native tool calls but content contains what looks like a JSON tool call
            if not raw_tool_calls and '{"name":' in content_text:
                log(f"[*] Fallback: Detecting tool calls in text content...", level="debug")
                # Look for {"name": "...", "arguments": {...}}
                matches = re.findall(r'(\{\s*"name":\s*".*?"\s*,\s*"arguments":\s*\{.*?\}\s*\})', content_text, re.DOTALL)
                for m in matches:
                    try:
                        parsed = json.loads(m)
                        if "name" in parsed and "arguments" in parsed:
                            raw_tool_calls.append({"function": parsed})
                            log(f"[*] Fallback: Parsed tool call '{parsed['name']}' from text.", level="verbose1")
                    except:
                        pass

            if not raw_tool_calls:
                break

            if response['message'].get('content') and not raw_tool_calls:
                log(f"\n{Color.paint(role_name, Color.BOLD + Color.CYAN)}: {response['message']['content']}\n", level="verbose2")
                response['message']['content'] = ""

            messages.append(response['message'])
            
            for tool_call in raw_tool_calls:
                tool_name = tool_call['function']['name']
                arguments = tool_call['function']['arguments']
                
                if tool_name == "trade_stock":
                    log(f"[*] Executing Trade: {arguments}", level="info")
                else:
                    log(f"[*] Calling tool: {tool_name}({arguments})", level="verbose1")
                
                try:
                    tool_result = await session.call_tool(tool_name, arguments)
                    # Correctly extract text from MCP content list
                    content = "".join([c.text for c in tool_result.content if hasattr(c, 'text')])
                    
                    if tool_name == "trade_stock":
                        log(f"[+] Trade Status: {content}", level="info")
                    else:
                        log(f"[+] Tool Result: {content}", level="verbose2")
                except Exception as e:
                    content = f"Error executing tool: {str(e)}"
                    log(f"Tool execution failed: {e}", level="error")
                
                messages.append({
                    'role': 'tool',
                    'content': content,
                })
            
            response = await llm_chat(messages, tools=ollama_tools, model=current_model)

        final_message = response['message']['content']
        if final_message:
            log(f"\n{Color.paint(role_name, Color.BOLD + Color.CYAN)}: {final_message}\n", level="verbose1")
        
        messages.append(response['message'])
        return messages

    finally:
        if spinner.task:
            await spinner.__aexit__(None, None, None)


async def llm_json_call(session, system_prompt, user_prompt, tools, schema_hint, model=None, role_name="AI"):
    """
    Call the LLM, let it use tools, then extract structured JSON from its final message.
    Retries once on parse failure with a correction prompt.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    messages = await process_ai_response(session, messages, tools, role_name=role_name, model=model)
    
    # Get the last assistant message
    last_msg = ""
    for m in reversed(messages):
        if m.get('role') == 'assistant' and m.get('content'):
            last_msg = m['content']
            break
    
    result = extract_json(last_msg)
    if result is not None:
        return result
    
    # Retry with correction prompt
    log(f"[{role_name}] JSON parse failed, retrying with correction...", level="verbose1")
    messages.append({
        "role": "user",
        "content": (
            f"Your previous response was not valid JSON. Please respond with ONLY valid JSON, "
            f"no commentary, no markdown. Match this schema:\n{schema_hint}"
        )
    })
    
    messages = await process_ai_response(session, messages, None, role_name=role_name, model=model)
    
    last_msg = ""
    for m in reversed(messages):
        if m.get('role') == 'assistant' and m.get('content'):
            last_msg = m['content']
            break
    
    result = extract_json(last_msg)
    return result

# ─────────────────────────────────────────────────────────────────────
# Phase Prompts
# ─────────────────────────────────────────────────────────────────────

PHASE_PROMPTS = {
    "intelligence": (
        "You are a market data collector for a stock trading system. Your ONLY job is to collect data, NOT to make trading decisions.\n\n"
        "Use the available tools IN THIS ORDER:\n"
        "1. Call `get_portfolio_summary` to get current cash and net worth\n"
        "2. Call `get_open_positions` to see what stocks we currently hold\n"
        "3. Call `get_market_movers` to see today's top gainers, losers, and most active\n\n"
        "After collecting all data, summarize what you found. Do NOT recommend any trades."
    ),
    "analyst": (
        "You are a stock analyst. Analyze the symbol {symbol} using the available tools.\n\n"
        "Call these tools:\n"
        "1. `get_ticker_details` for {symbol}\n"
        "2. `get_price_history` for {symbol}\n"
        "3. `get_ticker_news` for {symbol}\n"
        "4. `get_analyst_ratings` for {symbol}\n\n"
        "After collecting data, respond with ONLY a JSON object (no other text) matching this exact schema:\n"
        '{{\n'
        '  "symbol": "{symbol}",\n'
        '  "price": "<current price>",\n'
        '  "direction": "bullish" | "bearish" | "neutral",\n'
        '  "conviction": <1-10>,\n'
        '  "technical_summary": "<1-2 sentence summary of price action and trends>",\n'
        '  "catalysts": ["<catalyst 1>", "<catalyst 2>"],\n'
        '  "risk_factors": ["<risk 1>", "<risk 2>"],\n'
        '  "analyst_consensus": "<consensus recommendation or N/A>"\n'
        '}}'
    ),
    "strategist": (
        "You are a portfolio strategist. Your job is to decide what trades to make based on analysis reports.\n\n"
        "STRATEGY: {strategy} — {strategy_desc}\n"
        "RISK LEVEL: {risk_level} — Max position size: ${max_pos}\n"
        "COMMISSION: ${commission} per trade\n\n"
        "CURRENT PORTFOLIO:\n{portfolio_summary}\n\n"
        "CURRENT POSITIONS:\n{positions_summary}\n\n"
        "ANALYSIS REPORTS:\n{analysis_reports}\n\n"
        "RECENT TRADING HISTORY:\n{journal_summary}\n\n"
        "INSTRUCTIONS:\n"
        "- Consider what we already hold before buying more\n"
        "- Consider selling underperformers or taking profits\n"
        "- Respect the strategy and risk constraints\n"
        "- Factor in commission costs (${commission} per trade)\n"
        "- If no good opportunity exists, propose an empty list\n\n"
        "Respond with ONLY a JSON array (no other text). Each element:\n"
        '{{\n'
        '  "action": "Buy" | "Sell",\n'
        '  "symbol": "<TICKER>",\n'
        '  "quantity": <integer>,\n'
        '  "order_type": "Market" | "Limit",\n'
        '  "reasoning": "<1 sentence why>"\n'
        '}}\n\n'
        "If no trades are needed, respond with: []"
    ),
    "classic": (
        "You are a professional stock trader managing a portfolio on HowTheMarketWorks.\n\n"
        "STRATEGY: {strategy} — {strategy_desc}\n"
        "RISK LEVEL: {risk_level} — Max position: ${max_pos}, Cash buffer: {cash_buffer_pct}%\n"
        "COMMISSION: ${commission} per trade\n\n"
        "Your mission:\n"
        "1. Check your portfolio and market movers using the available tools\n"
        "2. Analyze promising symbols (price, news, analyst ratings)\n"
        "3. Execute trades using `trade_stock` if you find good opportunities\n"
        "4. Respect position size limits and cash buffer requirements\n\n"
        "Be decisive. If the data supports a trade, execute it."
    )
}

# ─────────────────────────────────────────────────────────────────────
# Risk Gate (Code-Based)
# ─────────────────────────────────────────────────────────────────────

def parse_currency(value):
    """Parse a currency string like '$1,234.56' into a float."""
    if isinstance(value, (int, float)):
        return float(value)
    if not value or value == "N/A":
        return 0.0
    try:
        cleaned = re.sub(r'[^\d.\-]', '', str(value))
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0

def risk_gate(proposals, portfolio, positions, risk_config):
    """
    Validate trade proposals against risk rules. Returns (approved, rejected) lists.
    This is deterministic Python code, NOT an LLM call.
    """
    cash = parse_currency(portfolio.get("cash", 0))
    buying_power = parse_currency(portfolio.get("buying_power", cash))
    net_worth = parse_currency(portfolio.get("net_worth", 0))
    max_pos = risk_config["max_pos"]
    cash_buffer = risk_config["cash_buffer"]
    commission = config["commission_rate"]
    
    # Calculate minimum cash to maintain
    min_cash = net_worth * cash_buffer if net_worth > 0 else cash * cash_buffer
    available_cash = max(0, cash - min_cash)
    
    # Build current position map
    position_map = {}
    for pos in positions:
        sym = pos.get("symbol", "").split('\n')[0].strip()
        val = parse_currency(pos.get("market_value", 0))
        position_map[sym] = val
    
    approved = []
    rejected = []
    running_spend = 0.0
    
    for trade in proposals:
        action = trade.get("action", "").strip()
        symbol = trade.get("symbol", "").strip().upper()
        quantity = int(trade.get("quantity", 0))
        reasoning = trade.get("reasoning", "")
        order_type = trade.get("order_type", "Market")
        
        if not symbol or quantity <= 0:
            rejected.append({**trade, "rejection_reason": "Invalid symbol or quantity"})
            continue
        
        if action == "Buy":
            # Estimate cost (rough — we don't know exact price, but we gatekeep on limits)
            existing_value = position_map.get(symbol, 0)
            
            # Check 1: Would the trade exceed max position size?
            # We can't know exact price here, so we check what we can
            if existing_value >= max_pos:
                rejected.append({**trade, "rejection_reason": f"Already at max position (${existing_value:.0f} >= ${max_pos})"})
                continue
            
            # Check 2: Do we have cash available (accounting for buffer)?
            trade_cost = commission  # At minimum we need to cover commission
            if available_cash - running_spend <= trade_cost:
                rejected.append({**trade, "rejection_reason": f"Insufficient cash after buffer (available: ${available_cash - running_spend:.0f})"})
                continue
            
            # Adjust quantity if position would exceed max
            # Rough estimate: cap spending at (max_pos - existing_value)
            max_spend = min(max_pos - existing_value, available_cash - running_spend - commission)
            if max_spend <= 0:
                rejected.append({**trade, "rejection_reason": "No room within position or cash limits"})
                continue
            
            running_spend += commission  # Track commission spend
            approved.append(trade)
            
        elif action == "Sell":
            # Check: Do we actually hold this stock?
            if symbol not in position_map and not any(
                p.get("symbol", "").split('\n')[0].strip() == symbol for p in positions
            ):
                rejected.append({**trade, "rejection_reason": f"Cannot sell {symbol} — not in portfolio"})
                continue
            
            approved.append(trade)
        else:
            rejected.append({**trade, "rejection_reason": f"Unknown action: {action}"})
    
    return approved, rejected

# ─────────────────────────────────────────────────────────────────────
# Pipeline Phases
# ─────────────────────────────────────────────────────────────────────

async def phase_intelligence(session, ollama_tools):
    """Phase 1: Gather market data using tools (Deterministic)."""
    log("Phase 1: Intelligence Gathering", level="phase")
    
    snapshot = {"portfolio": {}, "positions": [], "movers": {}, "raw_summary": "Data gathered deterministically."}
    
    # 1. Get Portfolio
    log("[*] Calling tool: get_portfolio_summary()", level="verbose1")
    try:
        res = await session.call_tool("get_portfolio_summary", {})
        text = "".join([c.text for c in res.content if hasattr(c, 'text')])
        snapshot["portfolio"] = extract_json(text) or {}
        log(f"[+] Tool Result: {text[:100]}...", level="verbose2")
    except Exception as e:
        log(f"Portfolio fetch failed: {e}", level="error")

    # 2. Get Positions
    log("[*] Calling tool: get_open_positions()", level="verbose1")
    try:
        res = await session.call_tool("get_open_positions", {})
        text = "".join([c.text for c in res.content if hasattr(c, 'text')])
        snapshot["positions"] = extract_json(text) or []
        log(f"[+] Tool Result: Found {len(snapshot['positions'])} positions", level="verbose2")
    except Exception as e:
        log(f"Positions fetch failed: {e}", level="error")

    # 3. Get Movers
    log("[*] Calling tool: get_market_movers()", level="verbose1")
    try:
        res = await session.call_tool("get_market_movers", {})
        text = "".join([c.text for c in res.content if hasattr(c, 'text')])
        snapshot["movers"] = extract_json(text) or {}
    except Exception as e:
        log(f"Movers fetch failed: {e}", level="error")
        
    # Log a better summary of movers
    movers_summary = []
    for cat, syms in snapshot["movers"].items():
        if isinstance(syms, list) and syms:
            sample = f" [{', '.join(syms[:3])}...]" if len(syms) > 3 else f" {syms}"
            movers_summary.append(f"{cat}{sample}({len(syms)})")
    
    log(f"[+] Movers Found: {', '.join(movers_summary) if movers_summary else 'None'}", level="verbose2")
    
    p_syms = [pos.get("symbol", "").split('\n')[0].strip().upper() for pos in snapshot["positions"]]
    log(f"  Portfolio: {snapshot['portfolio']}", level="verbose1")
    log(f"  Positions: {len(snapshot['positions'])} held{(' (' + ', '.join(p_syms) + ')') if p_syms else ''}", level="verbose1")
    log(f"  Movers: {len(snapshot['movers'])} categories found", level="verbose1")
    
    return snapshot

def extract_candidate_symbols(snapshot, watchlist):
    """Extract candidate symbols using a round-robin approach for category diversity."""
    import itertools
    candidates_set = set()
    
    # 1. Prepare lists for round-robin
    category_lists = []
    
    # From market movers categories
    for cat, syms in snapshot.get("movers", {}).items():
        if isinstance(syms, list) and syms:
            category_lists.append([str(s).strip().upper() for s in syms])
            
    # From watchlist
    if watchlist:
        category_lists.append([str(s).strip().upper() for s in watchlist])
        
    # From current positions (to check on holdings)
    pos_syms = [pos.get("symbol", "").split('\n')[0].strip().upper() for pos in snapshot.get("positions", [])]
    if pos_syms:
        category_lists.append(pos_syms)

    # 2. Blacklist of labels and common false positives
    blacklist = {
        "HTMW", "AI", "USD", "ETF", "BUY", "SELL", "HOLD", "NEWS", "SMA", "RSI", "ATH", "IPO", "SEC", "CEO", "CFO", "EPS", "GDP",
        "TFSA", "RRSP", "RESP", "FHSA", "LIRA", "RRIF", "RDSP", "TAX", "FUND", "ACCOUNT", "PORTFOLIO", "TOTAL", "CASH", "VALUE",
        "SYMBOL", "PRICE", "CHANGE", "CHG", "VOL", "VOLUME", "MKT", "CAP", "OPEN", "HIGH", "LOW", "DATE", "NEWS", "CHAT", "TFS"
    }

    # 3. Round-robin selection
    candidates = []
    for sym_tuple in itertools.zip_longest(*category_lists):
        for sym in sym_tuple:
            if sym and sym not in candidates_set and sym not in blacklist:
                if 1 <= len(sym) <= 5 and re.match(r'^[A-Z.\-]+$', sym):
                    candidates_set.add(sym)
                    candidates.append(sym)
    
    return candidates


async def _analyze_single_symbol(session, ollama_tools, symbol, schema_hint):
    """Analyze a single symbol. Used as a concurrent task in phase_analysis."""
    log(f"  ┌─ Analyzing {Color.paint(symbol, Color.BOLD + Color.YELLOW)}", level="info")
    
    system_prompt = PHASE_PROMPTS["analyst"].format(symbol=symbol)
    user_prompt = f"Analyze {symbol} now. Call all four tools, then provide your JSON analysis."
    
    result = await llm_json_call(
        session, system_prompt, user_prompt,
        tools=ollama_tools,
        schema_hint=schema_hint,
        model=config["agent_model"],
        role_name=f"Analyst({symbol})"
    )
    
    if result and isinstance(result, dict):
        result.setdefault("symbol", symbol)
        direction = result.get("direction", "?")
        conviction = result.get("conviction", "?")
        dir_color = Color.GREEN if direction == "bullish" else (Color.RED if direction == "bearish" else Color.YELLOW)
        log(f"  └─ {symbol}: {Color.paint(direction, dir_color)} (conviction: {conviction}/10)", level="info")
        return result
    else:
        log(f"  └─ {symbol}: Analysis failed (could not parse JSON)", level="info")
        return {"symbol": symbol, "direction": "neutral", "conviction": 3, "technical_summary": "Analysis unavailable", "catalysts": [], "risk_factors": ["Analysis failed"], "analyst_consensus": "N/A"}


async def phase_analysis(session, ollama_tools, candidates):
    """Phase 2: Deep-dive analysis on candidate symbols (runs concurrently)."""
    log("Phase 2: Symbol Analysis", level="phase")
    
    if not candidates:
        log("  No candidates to analyze.", level="info")
        return []
    
    symbols_to_analyze = candidates[:]
    log(f"  Analyzing {len(symbols_to_analyze)} symbol(s) concurrently: {', '.join(symbols_to_analyze)}", level="info")
    
    schema_hint = '{"symbol":"...","direction":"bullish|bearish|neutral","conviction":1-10,"technical_summary":"...","catalysts":[...],"risk_factors":[...],"analyst_consensus":"..."}'
    
    # Launch all analysts concurrently
    tasks = [
        _analyze_single_symbol(session, ollama_tools, symbol, schema_hint)
        for symbol in symbols_to_analyze
    ]
    reports = list(await asyncio.gather(*tasks))
    
    return reports


async def phase_strategy(session, snapshot, reports, journal_entries):
    """Phase 3: Decide what trades to make (pure reasoning, no tools)."""
    log("Phase 3: Trade Strategy", level="phase")
    
    strategy_desc = STRATEGY_MAP.get(config["strategy"], STRATEGY_MAP["DIVERSIFIED"])
    risk_data = RISK_MAP.get(config["risk_level"], RISK_MAP["MODERATE"])
    
    # Format data for the prompt
    portfolio_str = json.dumps(snapshot.get("portfolio", {}), indent=2)
    positions_str = json.dumps(snapshot.get("positions", []), indent=2) if snapshot.get("positions") else "No current positions."
    analysis_str = json.dumps(reports, indent=2) if reports else "No analysis reports available."
    
    # Format journal
    if journal_entries:
        journal_str = ""
        for entry in journal_entries[-3:]:
            journal_str += f"- Cycle {entry.get('cycle', '?')} ({entry.get('timestamp', '?')}): "
            trades = entry.get('trades_executed', [])
            if trades:
                journal_str += f"Executed {len(trades)} trade(s): {json.dumps(trades)}\n"
            else:
                journal_str += "No trades executed.\n"
    else:
        journal_str = "No previous trading history."
    
    system_prompt = PHASE_PROMPTS["strategist"].format(
        strategy=config["strategy"],
        strategy_desc=strategy_desc,
        risk_level=config["risk_level"],
        max_pos=risk_data["max_pos"],
        commission=config["commission_rate"],
        portfolio_summary=portfolio_str,
        positions_summary=positions_str,
        analysis_reports=analysis_str,
        journal_summary=journal_str
    )
    
    user_prompt = "Based on all the data above, propose your trades now. Respond with ONLY a JSON array."
    
    schema_hint = '[{"action":"Buy|Sell","symbol":"TICKER","quantity":10,"order_type":"Market","reasoning":"..."}]'
    
    result = await llm_json_call(
        session, system_prompt, user_prompt,
        tools=None,  # No tools — pure reasoning
        schema_hint=schema_hint,
        model=config["model"],
        role_name="Strategist"
    )
    
    if result is None:
        log("  Strategist returned no parseable proposals.", level="info")
        return []
    
    # Handle both single object and array responses
    if isinstance(result, dict):
        result = [result]
    
    if not isinstance(result, list):
        log("  Strategist response was not a list.", level="info")
        return []
    
    proposals = result
    log(f"  Proposed {len(proposals)} trade(s):", level="info")
    for p in proposals:
        action_color = Color.GREEN if p.get("action") == "Buy" else Color.RED
        log(f"    • {Color.paint(p.get('action', '?'), action_color)} {p.get('quantity', '?')} x {Color.paint(p.get('symbol', '?'), Color.BOLD)} — {p.get('reasoning', '')}", level="info")
    
    return proposals


async def phase_risk_gate(proposals, snapshot, positions):
    """Phase 4: Validate trades against risk rules (code-based)."""
    log("Phase 4: Risk Gate", level="phase")
    
    risk_data = RISK_MAP.get(config["risk_level"], RISK_MAP["MODERATE"])
    
    approved, rejected = risk_gate(proposals, snapshot.get("portfolio", {}), positions, risk_data)
    
    if rejected:
        log(f"  {Color.paint('REJECTED', Color.RED)} {len(rejected)} trade(s):", level="info")
        for r in rejected:
            log(f"    ✗ {r.get('action', '?')} {r.get('symbol', '?')}: {r.get('rejection_reason', 'Unknown')}", level="info")
    
    if approved:
        log(f"  {Color.paint('APPROVED', Color.GREEN)} {len(approved)} trade(s):", level="info")
        for a in approved:
            log(f"    ✓ {a.get('action', '?')} {a.get('quantity', '?')} x {a.get('symbol', '?')}", level="info")
    else:
        log("  No trades approved.", level="info")
    
    return approved, rejected


async def phase_execution(session, approved_trades, cycle_count, snapshot, reports):
    """Phase 5: Execute approved trades and journal the cycle."""
    log("Phase 5: Execution & Journaling", level="phase")
    
    executed = []
    
    for trade in approved_trades:
        symbol = trade["symbol"]
        action = trade["action"]
        quantity = trade["quantity"]
        order_type = trade.get("order_type", "Market")
        
        log(f"  Executing: {action} {quantity} x {symbol} ({order_type})...", level="info")
        
        try:
            result = await session.call_tool("trade_stock", {
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "order_type": order_type
            })
            status = str(result.content)
            log(f"  ✓ Result: {status}", level="info")
            executed.append({**trade, "status": status})
        except Exception as e:
            log(f"  ✗ Execution failed: {e}", level="error")
            executed.append({**trade, "status": f"FAILED: {e}"})
    
    # Build journal entry
    journal_entry = {
        "cycle": cycle_count,
        "timestamp": datetime.now().isoformat(),
        "portfolio_snapshot": snapshot.get("portfolio", {}),
        "symbols_analyzed": [r.get("symbol", "?") for r in reports],
        "trades_proposed": len(approved_trades),
        "trades_executed": executed,
    }
    
    save_journal_entry(journal_entry)
    log(f"  Journal entry saved for cycle #{cycle_count}.", level="verbose1")
    
    # Update watchlist with analyzed symbols
    watchlist = load_watchlist()
    for r in reports:
        sym = r.get("symbol", "")
        if sym and sym not in watchlist:
            watchlist.append(sym)
    save_watchlist(watchlist)
    
    return executed

# ─────────────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────────────

async def run_trader(auto_mode=False):
    server_params = StdioServerParameters(
        command=VENV_PYTHON,
        args=[MCP_SERVER_PATH],
        env=os.environ.copy()
    )

    async with AsyncExitStack() as stack:
        log(f"Connecting to MCP server at {MCP_SERVER_PATH}...", level="debug")
        results = await stack.enter_async_context(stdio_client(server_params))
        read, write = results
        session = await stack.enter_async_context(ClientSession(read, write))
        
        await session.initialize()
        
        tools_response = await session.list_tools()
        mcp_tools = tools_response.tools
        
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
        
        if auto_mode:
            banner = "\n--- HTMW AI Trader (AUTO MODE) Started ---"
            log(Color.paint(banner, Color.BOLD + Color.PURPLE), level="info")
            
            strategy_desc = STRATEGY_MAP.get(config["strategy"], "Diversified")
            risk_data = RISK_MAP.get(config["risk_level"], RISK_MAP["MODERATE"])
            
            config_summary = (
                f"  Strategy: {Color.paint(config['strategy'], Color.BOLD)} │ "
                f"Risk: {Color.paint(config['risk_level'], Color.BOLD)} │ "
                f"Max Pos: ${risk_data['max_pos']} │ "
                f"Buffer: {int(risk_data['cash_buffer']*100)}% │ "
                f"Commission: ${config['commission_rate']}"
            )
            log(config_summary, level="info")
            
            mode_label = "Native (5-Phase Pipeline)" if config["agentic_mode"] == "NATIVE" else "Classic (Single-Agent)"
            log(f"  Mode: {Color.paint(mode_label, Color.BOLD)} │ Model: {Color.paint(config['model'], Color.BOLD)}", level="info")
            log("Press Ctrl+C to stop.\n", level="info")
            
            cycle_count = 1
            while True:
                cycle_header = f"═══ Trade Cycle #{cycle_count} ═══"
                log(Color.paint(cycle_header, Color.BOLD + Color.YELLOW), level="info")
                
                if config["agentic_mode"] == "CLASSIC":
                    # ─── CLASSIC MODE ───
                    log("[AI] Running in Classic (Single-Agent) mode...", level="info")
                    risk_data = RISK_MAP.get(config["risk_level"], RISK_MAP["MODERATE"])
                    strategy_desc = STRATEGY_MAP.get(config["strategy"], STRATEGY_MAP["DIVERSIFIED"])
                    
                    classic_system = PHASE_PROMPTS["classic"].format(
                        strategy=config["strategy"],
                        strategy_desc=strategy_desc,
                        risk_level=config["risk_level"],
                        max_pos=risk_data["max_pos"],
                        cash_buffer_pct=int(risk_data["cash_buffer"] * 100),
                        commission=config["commission_rate"]
                    )
                    
                    messages = [
                        {"role": "system", "content": classic_system},
                        {"role": "user", "content": "Begin your analysis and trading cycle now."}
                    ]
                    messages = await process_ai_response(session, messages, ollama_tools, role_name="AI", model=config["model"])
                    
                    # Minimal journaling for classic mode
                    save_journal_entry({
                        "cycle": cycle_count,
                        "timestamp": datetime.now().isoformat(),
                        "mode": "classic",
                        "trades_executed": [],
                    })
                else:
                    # ─── NATIVE MODE (5-Phase Pipeline) ───
                    try:
                        # Phase 1: Intelligence
                        snapshot = await phase_intelligence(session, ollama_tools)
                        
                        # Extract candidates (code-based, not LLM)
                        watchlist = load_watchlist()
                        candidates = extract_candidate_symbols(snapshot, watchlist)
                        
                        if not candidates:
                            log("\n  No candidate symbols identified. Skipping this cycle.", level="info")
                            save_journal_entry({
                                "cycle": cycle_count,
                                "timestamp": datetime.now().isoformat(),
                                "note": "No candidates found",
                                "trades_executed": [],
                            })
                        else:
                            # Phase 2: Analysis
                            reports = await phase_analysis(session, ollama_tools, candidates)
                            
                            # Phase 3: Strategy
                            journal_entries = load_journal()
                            proposals = await phase_strategy(session, snapshot, reports, journal_entries)
                            
                            if not proposals:
                                log("\n  Strategist recommends holding. No trades this cycle.", level="info")
                                save_journal_entry({
                                    "cycle": cycle_count,
                                    "timestamp": datetime.now().isoformat(),
                                    "symbols_analyzed": [r.get("symbol") for r in reports],
                                    "note": "Strategist recommended holding",
                                    "trades_executed": [],
                                })
                            else:
                                # Phase 4: Risk Gate
                                approved, rejected = await phase_risk_gate(proposals, snapshot, snapshot.get("positions", []))
                                
                                if approved:
                                    # Phase 5: Execute
                                    await phase_execution(session, approved, cycle_count, snapshot, reports)
                                else:
                                    log("\n  All proposals rejected by Risk Gate.", level="info")
                                    save_journal_entry({
                                        "cycle": cycle_count,
                                        "timestamp": datetime.now().isoformat(),
                                        "symbols_analyzed": [r.get("symbol") for r in reports],
                                        "trades_proposed": len(proposals),
                                        "note": "All trades rejected by risk gate",
                                        "trades_executed": [],
                                    })
                    
                    except Exception as e:
                        log(f"Pipeline error in cycle #{cycle_count}: {e}", level="error")
                        import traceback
                        log(traceback.format_exc(), level="debug")
                
                log(f"\n{'─'*40}", level="info")
                log(f"Cycle #{cycle_count} complete. Waiting 15 minutes...\n", level="info")
                cycle_count += 1
                await asyncio.sleep(900)
        else:
            # ─── INTERACTIVE MODE ───
            log("\n--- HTMW AI Trader (Ollama + MCP) Ready ---", level="info")
            log("Type 'exit' to quit.\n", level="info")

            messages = [
                {"role": "system", "content": "You are a professional stock trading assistant. You have access to the HowTheMarketWorks trading platform through various tools. Use them to help the user manage their portfolio, check news, and execute trades."}
            ]

            while True:
                try:
                    user_input = input("You: ")
                except EOFError:
                    break
                    
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                messages.append({"role": "user", "content": user_input})
                messages = await process_ai_response(session, messages, ollama_tools)
                
                # Keep history manageable
                if len(messages) > 20:
                    messages = [messages[0]] + messages[-18:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTMW AI Trader with Ollama")
    parser.add_argument("--auto", action="store_true", help="Run in automatic trading mode")
    parser.add_argument("--verbose", type=int, nargs="?", const=1, default=0, help="Verbose level: 0=regular, 1=tool calls, 2=tool results")
    parser.add_argument("--silent", action="store_true", help="Suppress all output except errors")
    parser.add_argument("--debug", action="store_true", help="Show internal debug messages and tool traces")
    parser.add_argument("--colorize", action="store_true", help="Enable colorized output")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

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
