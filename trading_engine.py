import asyncio
import os
import json
import time
import sys
import re
import random
import itertools
from datetime import datetime
from contextlib import AsyncExitStack
from typing import Optional, List, Dict, Any

# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# import ollama
# from openai import AsyncOpenAI
from dotenv import load_dotenv

# Interface Integration
class EventRegistry:
    def __init__(self):
        self.queues = []

    def subscribe(self):
        q = asyncio.Queue()
        self.queues.append(q)
        return q

    def unsubscribe(self, q):
        if q in self.queues:
            self.queues.remove(q)

    def post_event(self, event_type: str, data: Any):
        event = {"type": event_type, "data": data}
        for q in self.queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

events = EventRegistry()

class TradeEventHandler:
    """Helper to route events to any subscribed interfaces."""
    @staticmethod
    def status(text):
        events.post_event("status", text)
        
    @staticmethod
    def ai_log(text):
        events.post_event("ai_log", text)
        
    @staticmethod
    def system_log(text):
        events.post_event("system_log", text)
        
    @staticmethod
    def positions(data):
        events.post_event("positions", data)
        
    @staticmethod
    def watchlist(data):
        events.post_event("watchlist", data)
        
    @staticmethod
    def portfolio_value(data):
        events.post_event("portfolio_value", data)

    @staticmethod
    def portfolio_snapshot(data):
        events.post_event("portfolio_snapshot", data)

    @staticmethod
    def performance(data):
        events.post_event("performance", data)

load_dotenv()

# Configuration
DEFAULT_MODEL = "gpt-oss:20b"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_SERVER_PATH = os.path.join(PROJECT_DIR, "mcp_scrape.py")
VENV_PYTHON = os.path.join(PROJECT_DIR, ".venv/bin/python")
JOURNAL_PATH = os.path.join(PROJECT_DIR, "config", "trading_journal.json")
WATCHLIST_PATH = os.path.join(PROJECT_DIR, "config", "watchlist.json")
MEMORY_PATH = os.path.join(PROJECT_DIR, "config", "memory.json")

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
    "commission_rate": float(os.getenv("COMMISSION_RATE", "0")),
    "model": os.getenv("LLM_MODEL", DEFAULT_MODEL)
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
        if events.queues: return # No spinner if an interface is attached
        
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
        if events.queues: return self
        if config["verbose_level"] == 0 and not config["silent"] and not config["debug"]:
            self.task = asyncio.create_task(self.spin())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.task:
            self.stop_event.set()
            await self.task

def log(msg, level="info"):
    # ALWAYS route to events if subscribers exist
    if events.queues:
        if level == "error":
            TradeEventHandler.system_log(f"[red]ERROR:[/red] {msg}")
        elif level == "phase":
            TradeEventHandler.status(msg)
        elif level == "verbose1" or level == "info":
             TradeEventHandler.system_log(msg)
        elif level == "verbose2":
             TradeEventHandler.system_log(f"[dim]{msg}[/dim]")
        # Do NOT print to stdout in TUI mode
        return

    if config["silent"]:
        if level == "error":
            print(f"ERROR: {msg}", file=sys.stderr)
        return

    if level == "debug" and not config["debug"]:
        return
    
    if level == "error":
        print(f"{Color.paint('ERROR:', Color.RED)} {msg}", file=sys.stderr, flush=True)
    elif level == "phase":
        print(f"\n{Color.paint('───', Color.DIM)} {Color.paint(msg, Color.BOLD + Color.PURPLE)} {Color.paint('───', Color.DIM)}", flush=True)
    else:
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

def extract_json(text):
    if not text or not text.strip(): return None
    text = text.strip()
    try: return json.loads(text)
    except json.JSONDecodeError: pass
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if code_block_match:
        try: return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError: pass
    brace_depth = 0
    start = -1
    for i, c in enumerate(text):
        if c == '{':
            if brace_depth == 0: start = i
            brace_depth += 1
        elif c == '}':
            brace_depth -= 1
            if brace_depth == 0 and start != -1:
                try: return json.loads(text[start:i+1])
                except json.JSONDecodeError: start = -1
    bracket_depth = 0
    start = -1
    for i, c in enumerate(text):
        if c == '[':
            if bracket_depth == 0: start = i
            bracket_depth += 1
        elif c == ']':
            bracket_depth -= 1
            if bracket_depth == 0 and start != -1:
                try: return json.loads(text[start:i+1])
                except json.JSONDecodeError: start = -1
    return None

def load_journal(limit=20):
    if not os.path.exists(JOURNAL_PATH): return []
    try:
        with open(JOURNAL_PATH, 'r') as f:
            entries = json.load(f)
        return entries[-limit:]
    except: return []

def save_journal_entry(entry):
    entries = []
    if os.path.exists(JOURNAL_PATH):
        try:
            with open(JOURNAL_PATH, 'r') as f: entries = json.load(f)
        except: entries = []
    entries.append(entry)
    entries = entries[-100:]
    with open(JOURNAL_PATH, 'w') as f: json.dump(entries, f, indent=2)

def load_watchlist():
    if not os.path.exists(WATCHLIST_PATH): return []
    try:
        with open(WATCHLIST_PATH, 'r') as f: return json.load(f)
    except: return []

def save_watchlist(symbols):
    symbols = list(dict.fromkeys(symbols))[:50]
    with open(WATCHLIST_PATH, 'w') as f: json.dump(symbols, f, indent=2)

def load_memory():
    if not os.path.exists(MEMORY_PATH): return {"learnings": [], "last_cycle_summary": "First cycle."}
    try:
        with open(MEMORY_PATH, 'r') as f: return json.load(f)
    except: return {"learnings": [], "last_cycle_summary": "Memory corrupted."}

def save_memory(memory):
    with open(MEMORY_PATH, 'w') as f: json.dump(memory, f, indent=2)


def calculate_performance_stats():
    journal = load_journal(limit=100)
    if not journal: return {"win_rate": 0, "roi": 0, "profit_factor": 0}
    
    trades = []
    for entry in journal:
        for t in entry.get("trades_executed", []): trades.append(t)
    
    # Very simplified ROI since we don't have historical net worth easily aggregated here
    # We use current vs start if available
    first_snapshot = journal[0].get("portfolio_snapshot", {})
    last_snapshot = journal[-1].get("portfolio_snapshot", {})
    
    start_val = parse_currency(first_snapshot.get("net_worth", 0))
    end_val = parse_currency(last_snapshot.get("net_worth", 0))
    roi = ((end_val - start_val) / start_val * 100) if start_val > 0 else 0
    
    # Win rate based on net worth change per cycle (approximation)
    wins = 0
    total_cycles = len(journal)
    for i in range(1, total_cycles):
        prev = parse_currency(journal[i-1].get("portfolio_snapshot", {}).get("net_worth", 0))
        curr = parse_currency(journal[i].get("portfolio_snapshot", {}).get("net_worth", 0))
        if curr > prev: wins += 1
    
    win_rate = (wins / (total_cycles - 1) * 100) if total_cycles > 1 else 0
    
    return {
        "win_rate": round(win_rate, 2),
        "roi": round(roi, 2),
        "profit_factor": 1.5, # Placeholder for more complex math
        "total_trades": len(trades)
    }

# LLM Provider Global
_llm_provider = None

def set_llm_provider(provider):
    global _llm_provider
    _llm_provider = provider

async def llm_chat(messages, tools=None, model=None):
    if not _llm_provider:
        raise ValueError("LLM Provider not set. Call set_llm_provider() first.")
    return await _llm_provider.chat(messages, tools=tools, model=model)

async def process_ai_response(session, messages, ollama_tools, role_name="AI", model=None):
    current_model = model or config["model"]
    log(f"[{role_name}] Calling {config['provider']} with model {current_model}...", level="debug")
    async with AISpinner():
        response = await llm_chat(messages, tools=ollama_tools, model=current_model)
        while response.get('message', {}).get('tool_calls') or (response['message'].get('content') and '{"name":' in response['message'].get('content')):
            raw_tool_calls = response['message'].get('tool_calls') or []
            content_text = response['message'].get('content', "")
            if not raw_tool_calls and '{"name":' in content_text:
                matches = re.findall(r'(\{\s*"name":\s*".*?"\s*,\s*"arguments":\s*\{.*?\}\s*\})', content_text, re.DOTALL)
                for m in matches:
                    try:
                        parsed = json.loads(m)
                        if "name" in parsed and "arguments" in parsed: raw_tool_calls.append({"function": parsed})
                    except: pass
            if not raw_tool_calls: break
            messages.append(response['message'])
            for tool_call in raw_tool_calls:
                tool_name = tool_call['function']['name']
                arguments = tool_call['function']['arguments']
                log(f"[*] Calling tool: {tool_name}({arguments})", level="verbose1")
                try:
                    tool_result = await session.call_tool(tool_name, arguments)
                    content = "".join([c.text for c in tool_result.content if hasattr(c, 'text')])
                    log(f"[+] Tool Result: {content[:100]}...", level="verbose2")
                except Exception as e:
                    content = f"Error executing tool: {str(e)}"
                    log(f"Tool execution failed: {e}", level="error")
                messages.append({'role': 'tool', 'content': content})
            response = await llm_chat(messages, tools=ollama_tools, model=current_model)
        final_message = response['message']['content']
        if final_message: log(f"\n{Color.paint(role_name, Color.BOLD + Color.CYAN)}: {final_message}\n", level="verbose1")
        messages.append(response['message'])
        return messages

async def llm_json_call(session, system_prompt, user_prompt, tools, schema_hint, model=None, role_name="AI"):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    messages = await process_ai_response(session, messages, tools, role_name=role_name, model=model)
    last_msg = next((m['content'] for m in reversed(messages) if m.get('role') == 'assistant' and m.get('content')), "")
    result = extract_json(last_msg)
    if result is not None: return result
    log(f"[{role_name}] JSON parse failed, retrying...", level="verbose1")
    messages.append({"role": "user", "content": f"Your previous response was not valid JSON. Match this schema:\n{schema_hint}"})
    messages = await process_ai_response(session, messages, None, role_name=role_name, model=model)
    last_msg = next((m['content'] for m in reversed(messages) if m.get('role') == 'assistant' and m.get('content')), "")
    return extract_json(last_msg)

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
        "4. `get_analyst_ratings` for {symbol}\n"
        "Optional: Use `web_search` and `scrape_url` for broader sentiment analysis if needed.\n\n"
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
        "You also have access to `web_search` and `scrape_url` for general market research and competitor analysis outside HTMW.\n\n"
        "Be decisive. If the data supports a trade, execute it."
    )
}

def parse_currency(value):
    if isinstance(value, (int, float)): return float(value)
    if not value or value == "N/A": return 0.0
    cleaned = re.sub(r'[^\d.\-]', '', str(value))
    return float(cleaned) if cleaned else 0.0

def risk_gate(proposals, portfolio, positions, risk_config):
    cash = parse_currency(portfolio.get("cash", 0))
    net_worth = parse_currency(portfolio.get("net_worth", 0))
    min_cash = net_worth * risk_config["cash_buffer"]
    available_cash = max(0, cash - min_cash)
    position_map = {p.get("symbol", "").split('\n')[0].strip(): parse_currency(p.get("market_value", 0)) for p in positions}
    approved, rejected = [], []
    running_spend = 0.0
    for trade in proposals:
        action, symbol, quantity = trade.get("action", ""), trade.get("symbol", "").upper(), int(trade.get("quantity", 0))
        if not symbol or quantity <= 0: rejected.append({**trade, "rejection_reason": "Invalid symbol/qty"}); continue
        if action == "Buy":
            if position_map.get(symbol, 0) >= risk_config["max_pos"]: rejected.append({**trade, "rejection_reason": "Max position exceeded"}); continue
            trade_cost = config["commission_rate"]
            if available_cash - running_spend <= trade_cost: rejected.append({**trade, "rejection_reason": "Insufficient cash"}); continue
            running_spend += trade_cost; approved.append(trade)
        elif action == "Sell":
            if symbol not in position_map: rejected.append({**trade, "rejection_reason": "Not in portfolio"}); continue
            approved.append(trade)
        else: rejected.append({**trade, "rejection_reason": f"Unknown action: {action}"})
    return approved, rejected

async def phase_intelligence(session, ollama_tools):
    log("Phase 1: Intelligence Gathering", level="phase")
    snapshot = {"portfolio": {}, "positions": [], "movers": {}}
    try:
        res = await session.call_tool("get_portfolio_summary", {})
        snapshot["portfolio"] = extract_json("".join([c.text for c in res.content if hasattr(c, 'text')])) or {}
        TradeEventHandler.portfolio_snapshot(snapshot["portfolio"])
    except Exception as e: log(f"Portfolio failed: {e}", level="error")
    try:
        res = await session.call_tool("get_open_positions", {})
        snapshot["positions"] = extract_json("".join([c.text for c in res.content if hasattr(c, 'text')])) or []
        TradeEventHandler.positions(snapshot['positions'])
    except Exception as e: log(f"Positions failed: {e}", level="error")
    try:
        res = await session.call_tool("get_market_movers", {})
        snapshot["movers"] = extract_json("".join([c.text for c in res.content if hasattr(c, 'text')])) or {}
    except Exception as e: log(f"Movers failed: {e}", level="error")
    
    # Portfolio history for graph
    journal = load_journal()
    val_history = [parse_currency(e.get("portfolio_snapshot", {}).get("net_worth", 0)) for e in journal[-20:]]
    val_history.append(parse_currency(snapshot['portfolio'].get('net_worth', 0)))
    TradeEventHandler.portfolio_value([v for v in val_history if v > 0])
    return snapshot

def extract_candidate_symbols(snapshot, watchlist):
    cat_lists = [syms for cat, syms in snapshot.get("movers", {}).items() if isinstance(syms, list)]
    if watchlist: cat_lists.append(watchlist)
    cat_lists.append([p.get("symbol", "").split('\n')[0].strip().upper() for p in snapshot.get("positions", [])])
    blacklist = {"TFSA","USD","ETF","BUY","SELL","CASH","TOTAL","VALUE","SYMBOL","PRICE"}
    candidates = []
    for sym_tuple in itertools.zip_longest(*cat_lists):
        for sym in sym_tuple:
            if sym and str(sym).upper() not in candidates and str(sym).upper() not in blacklist and 1 <= len(str(sym)) <= 5:
                candidates.append(str(sym).upper())
    return candidates

async def _analyze_single_symbol(session, ollama_tools, symbol, schema_hint):
    log(f"  ┌─ Analyzing {symbol}", level="info")
    result = await llm_json_call(session, PHASE_PROMPTS["analyst"].format(symbol=symbol), f"Analyze {symbol} now.", ollama_tools, schema_hint, model=config["agent_model"], role_name=f"Analyst({symbol})")
    if result and isinstance(result, dict):
        result.setdefault("symbol", symbol)
        TradeEventHandler.ai_log(f"[{symbol}] {result.get('direction','?').upper()} ({result.get('conviction','?')}/10): {result.get('technical_summary','')}")
        return result
    return {"symbol": symbol, "direction": "neutral", "conviction": 3}

async def phase_analysis(session, ollama_tools, candidates):
    log("Phase 2: Alpha Analysis", level="phase")
    if not candidates: return []
    symbols = candidates[:10]
    schema_hint = '{"symbol":"...","direction":"bullish|bearish|neutral","conviction":1-10,"technical_summary":"..."}'
    tasks = [_analyze_single_symbol(session, ollama_tools, s, schema_hint) for s in symbols]
    return list(await asyncio.gather(*tasks))

async def phase_strategy(session, snapshot, reports, journal, memory):
    log("Phase 3: Tactical Strategy", level="phase")
    system = PHASE_PROMPTS["strategist"].format(strategy=config["strategy"], strategy_desc=STRATEGY_MAP.get(config["strategy"],""), risk_level=config["risk_level"], max_pos=RISK_MAP.get(config["risk_level"],{}).get("max_pos",0), commission=config["commission_rate"], portfolio_summary=json.dumps(snapshot["portfolio"]), positions_summary=json.dumps(snapshot["positions"]), analysis_reports=json.dumps(reports), journal_summary=json.dumps(journal[-3:]))
    
    # Inject memory
    system += f"\n\nPREVIOUS CYCLE MEMORY:\n{memory.get('last_cycle_summary', 'No memory.')}"
    
    schema_hint = '[{"action":"Buy|Sell","symbol":"TICKER","quantity":10,"reasoning":"..."}]'
    proposals = await llm_json_call(session, system, "Propose trades.", None, schema_hint, role_name="Strategist")
    if proposals:
        if isinstance(proposals, dict): proposals = [proposals]
        TradeEventHandler.ai_log(f"[Strategy] {proposals[0].get('reasoning','') or 'Proposing trades.'}")
        return proposals
    return []

async def phase_post_cycle(session, snapshot, approved, memory):
    log("Phase 4: Reflection & Memory", level="phase")
    prompt = (
        "Briefly summarize this cycle's actions and outcomes for your future self.\n"
        f"Portfolio: {snapshot['portfolio'].get('net_worth')}\n"
        f"Trades: {json.dumps(approved)}\n"
        "What should we remember for next time?"
    )
    res = await llm_chat([{"role": "user", "content": prompt}], model=config["agent_model"])
    summary = res['message']['content'][:500]
    memory["last_cycle_summary"] = summary
    save_memory(memory)
    log(f"Memory Updated: {summary[:100]}...", level="verbose2")

async def run_trader(auto_mode=False, provider=None):
    if provider: set_llm_provider(provider)
    
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    
    env = os.environ.copy()
    env["FASTMCP_BANNER"] = "0"
    server_params = StdioServerParameters(command=VENV_PYTHON, args=[MCP_SERVER_PATH], env=env)
    async with AsyncExitStack() as stack:
        TradeEventHandler.status("Connecting to MCP...")
        read, write = await stack.enter_async_context(stdio_client(server_params))
        
        TradeEventHandler.status("Initializing Session...")
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        
        TradeEventHandler.status("Listing Tools...")
        tools_res = await session.list_tools()
        ollama_tools = [{'type': 'function', 'function': {'name': t.name, 'description': t.description, 'parameters': t.inputSchema}} for t in tools_res.tools]
        log(f"Connected to MCP Server. Found {len(ollama_tools)} tools.", level="info")
        TradeEventHandler.status("Ready")
        
        if not auto_mode:
            log("Interactive mode ready.", level="info")
            msg = [{"role": "system", "content": "You are a trading assistant."}]
            while True:
                try: user_input = input("You: ")
                except: break
                if user_input.lower() in ["exit","quit"]: break
                msg.append({"role": "user", "content": user_input})
                msg = await process_ai_response(session, msg, ollama_tools)
        else:
            cycle = 1
            memory = load_memory()
            while True:
                log(f"Trade Cycle #{cycle}", level="phase")
                try:
                    # Analytics update
                    TradeEventHandler.performance(calculate_performance_stats())
                    
                    snapshot = await phase_intelligence(session, ollama_tools)
                    candidates = extract_candidate_symbols(snapshot, load_watchlist())
                    reports = await phase_analysis(session, ollama_tools, candidates)
                    proposals = await phase_strategy(session, snapshot, reports, load_journal(), memory)
                    approved, _ = risk_gate(proposals, snapshot["portfolio"], snapshot["positions"], RISK_MAP.get(config["risk_level"],RISK_MAP["MODERATE"]))
                    
                    for trade in approved:
                        log(f"Executing: {trade['action']} {trade['quantity']} {trade['symbol']}", level="info")
                        try:
                            res = await session.call_tool("trade_stock", {"symbol": trade["symbol"], "action": trade["action"], "quantity": trade["quantity"]})
                            TradeEventHandler.system_log(f"[green]Trade Success: {trade['symbol']}[/green]")
                        except Exception as e: log(f"Trade failed: {e}", level="error")
                    
                    save_journal_entry({"cycle": cycle, "timestamp": datetime.now().isoformat(), "portfolio_snapshot": snapshot["portfolio"], "trades_executed": approved})
                    
                    # Reflection
                    await phase_post_cycle(session, snapshot, approved, memory)
                    
                except Exception as e: log(f"Cycle error: {e}", level="error")
                log("Cycle complete. Waiting...", level="info")
                cycle += 1
                await asyncio.sleep(900)
