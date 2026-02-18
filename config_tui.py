import os
import sys
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme
from dotenv import load_dotenv, set_key

# Load existing .env
load_dotenv()

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green",
    "highlight": "bold magenta"
})

console = Console(theme=custom_theme)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_banner():
    banner = Text(r"""
    █ █ ▀█▀ █▀▄▀█ █     ▄▀█ █ 
    █▀█  █  █ ▀ █ ▀ ▄   █▀█ █ 
    """, style="highlight")
    console.print(Panel(banner, title="[bold white]HTMW Trader Configuration[/bold white]", border_style="highlight"))

def save_setting(key, value):
    env_path = os.path.join(os.getcwd(), '.env')
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write("")
    set_key(env_path, key, value)

def manage_credentials():
    console.print("\n[bold cyan]HTMW Credentials[/bold cyan]")
    username = questionary.text("Username:", default=os.getenv("USERNAME", "")).ask()
    password = questionary.password("Password:", default=os.getenv("PASSWORD", "")).ask()
    
    if username is not None: save_setting("USERNAME", username)
    if password is not None: save_setting("PASSWORD", password)
    console.print("[success]Credentials saved![/success]")

def manage_provider():
    console.print("\n[bold cyan]LLM Provider Settings[/bold cyan]")
    provider = questionary.select(
        "Select Provider:",
        choices=["ollama", "openai-compatible (llama.cpp, vLLM, etc.)"],
        default=os.getenv("LLM_PROVIDER", "ollama")
    ).ask()
    
    if provider == "openai-compatible (llama.cpp, vLLM, etc.)":
        provider = "openai"
    
    save_setting("LLM_PROVIDER", provider)
    
    base_url = questionary.text(
        "Base URL:",
        default=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1" if provider == "ollama" else "http://localhost:8000/v1")
    ).ask()
    save_setting("LLM_BASE_URL", base_url)
    
    api_key = questionary.text("API Key (use 'ollama' for local):", default=os.getenv("LLM_API_KEY", "ollama")).ask()
    save_setting("LLM_API_KEY", api_key)
    
    console.print("[success]Provider settings saved![/success]")

def manage_models():
    console.print("\n[bold cyan]Model Selection[/bold cyan]")
    main_model = questionary.text("Main Model (Lead PM):", default=os.getenv("LLM_MODEL", "gpt-oss:20b")).ask()
    save_setting("LLM_MODEL", main_model)
    
    agent_model = questionary.text("Agent Model (Specialists):", default=os.getenv("AGENT_MODEL", main_model)).ask()
    save_setting("AGENT_MODEL", agent_model)
    
    console.print("[success]Models configured![/success]")

def manage_trading():
    console.print("\n[bold cyan]Trading Strategy[/bold cyan]")
    strategy = questionary.select(
        "Select Strategy:",
        choices=["DIVERSIFIED", "MOMENTUM", "MEAN_REVERSION", "CATALYST"],
        default=os.getenv("TRADING_STRATEGY", "DIVERSIFIED")
    ).ask()
    save_setting("TRADING_STRATEGY", strategy)
    
    risk_level = questionary.select(
        "Select Risk Level:",
        choices=["CONSERVATIVE", "MODERATE", "AGGRESSIVE"],
        default=os.getenv("RISK_LEVEL", "MODERATE")
    ).ask()
    save_setting("RISK_LEVEL", risk_level)
    
    console.print("[success]Trading parameters set![/success]")

def manage_agent_settings():
    console.print("\n[bold cyan]Agent Settings[/bold cyan]")
    mode = questionary.select(
        "Agentic Mode:",
        choices=["Native (Multi-Agent Council)", "Classic (Single-Agent)"],
        default="Native (Multi-Agent Council)" if os.getenv("AGENTIC_MODE", "NATIVE") == "NATIVE" else "Classic (Single-Agent)"
    ).ask()
    save_setting("AGENTIC_MODE", "NATIVE" if "Native" in mode else "CLASSIC")
    
    console.print("[success]Agent settings updated![/success]")

def main_menu():
    while True:
        clear_screen()
        show_banner()
        
        choice = questionary.select(
            "Main Menu:",
            choices=[
                "1. Configure HTMW Credentials",
                "2. Configure LLM Provider & Base URL",
                "3. Configure Models (Main vs Specialists)",
                "4. Configure Strategy & Risk",
                "5. Configure Agent Mode (Single vs Multi)",
                "6. Launch Trader",
                "7. Exit"
            ]
        ).ask()
        
        if choice == "1. Configure HTMW Credentials":
            manage_credentials()
        elif choice == "2. Configure LLM Provider & Base URL":
            manage_provider()
        elif choice == "3. Configure Models (Main vs Specialists)":
            manage_models()
        elif choice == "4. Configure Strategy & Risk":
            manage_trading()
        elif choice == "5. Configure Agent Mode (Single vs Multi)":
            manage_agent_settings()
        elif choice == "6. Launch Trader":
            console.print("\n[bold yellow]Launching Trader...[/bold yellow]")
            os.execvp(sys.executable, [sys.executable, "ollama_trader.py", "--auto", "--colorize"])
        elif choice == "7. Exit":
            console.print("\n[bold green]Goodbye![/bold green]")
            sys.exit(0)
        
        key = questionary.press_any_key_to_continue().ask()

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[warning]Configuration cancelled.[/warning]")
        sys.exit(0)
