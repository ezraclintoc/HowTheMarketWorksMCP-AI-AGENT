from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
from dotenv import load_dotenv
import os
import json
import pickle
import logging
from fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HTMWScraper")

class HTMWTrader:
    def __init__(self, username, password, headless=True, verbose=False, cookie_path="htmw_cookies.pkl"):
        self.username = username
        self.password = password
        self.headless = headless
        self.verbose = verbose
        self.cookie_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cookie_path)
        self.driver = None
        self.ensure_driver()

    def _setup_driver(self):
        if self.verbose: logger.info("Setting up new Chrome driver...")
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver

    def ensure_driver(self):
        """Checks if driver is alive, restarts if needed."""
        try:
            if self.driver:
                # Try a lightweight check
                self.driver.current_url
                return True
        except Exception:
            if self.verbose: logger.warning("Driver is dead or not initialized. Reconnecting...")
            try: self.driver.quit()
            except: pass
            
        self.driver = self._setup_driver()
        self.login()
        return True

    def save_cookies(self):
        try:
            with open(self.cookie_path, 'wb') as f:
                pickle.dump(self.driver.get_cookies(), f)
            if self.verbose: logger.info("Cookies saved successfully.")
        except Exception as e:
            if self.verbose: logger.error(f"Failed to save cookies: {e}")

    def load_cookies(self):
        if not os.path.exists(self.cookie_path):
            return False
        try:
            with open(self.cookie_path, 'rb') as f:
                cookies = pickle.load(f)
                for cookie in cookies:
                    self.driver.add_cookie(cookie)
            if self.verbose: logger.info("Cookies loaded successfully.")
            return True
        except Exception as e:
            if self.verbose: logger.error(f"Failed to load cookies: {e}")
            return False

    def login(self):
        try:
            self.driver.get("https://app.howthemarketworks.com/login")
            
            # Try loading cookies first
            if self.load_cookies():
                self.driver.refresh()
                try:
                    # Check if we are actually logged in by looking for a logged-in element (e.g., .summary)
                    WebDriverWait(self.driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".summary")))
                    if self.verbose: logger.info("Logged in via cookies.")
                    return
                except:
                    if self.verbose: logger.info("Cookies expired or invalid. Proceeding with full login.")
            
            wait = WebDriverWait(self.driver, 15)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".login-register-form")))

            # Handle Cookie Banner (if present)
            try:
                cookie_btn = self.driver.find_element(By.ID, "cookie-banner-btn")
                cookie_btn.click()
                if self.verbose: logger.info("Closed cookie banner.")
                time.sleep(1)
            except:
                pass

            # Find and fill username
            user_field = self.driver.find_element(By.ID, "UserName")
            user_field.clear()
            user_field.send_keys(self.username)
            
            # Find and fill password
            pass_field = self.driver.find_element(By.ID, "Password")
            pass_field.clear()
            pass_field.send_keys(self.password)
            
            # Click Login
            login_btn = self.driver.find_element(By.XPATH, "//input[@value='Login']")
            login_btn.click()
            
            if self.verbose: logger.info("Login credentials submitted.")
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".summary"))) # Wait for login to process
            
            # Save cookies after successful login
            self.save_cookies()
            
        except Exception as e:
            if self.verbose: logger.error(f"Error during login: {e}")
            raise e

    def get_open_positions(self):
        self.ensure_driver()
        try:
            self.driver.get("https://app.howthemarketworks.com/accounting/openpositions")
            wait = WebDriverWait(self.driver, 10)
            
            rows = []
            try:
                # Target the specific equity table and data body identified in debugging
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table#tOpenPositions_equities tbody.openpositions-data tr")))
                rows = self.driver.find_elements(By.CSS_SELECTOR, "table#tOpenPositions_equities tbody.openpositions-data tr")
            except Exception as e:
                if self.verbose: print(f"Open positions table not found: {e}")
            
            positions = []
            for row in rows:
                cols = row.find_elements(By.TAG_NAME, "td")
                # HTMW column mapping: 0=Symbol, 3=Quantity, 4=Price Paid, 6=Market Value
                if len(cols) > 6:
                    symbol_full = cols[0].text.strip()
                    symbol = symbol_full.split('\n')[0] # Remove descriptive text if present
                    
                    positions.append({
                        "symbol": symbol,
                        "quantity": cols[3].text.strip(),
                        "price": cols[4].text.strip(),
                        "market_value": cols[6].text.strip(),
                        "status": "Filled"
                    })
                    
            # Handle Pending Orders by merging them into positions
            try:
                pending = self.get_pending_orders()
                for p in pending:
                    # Avoid duplicates if HTMW somehow lists them in both (unlikely)
                    if not any(pos['symbol'] == p['symbol'] and pos['status'] == 'Filled' for pos in positions):
                        positions.append({
                            "symbol": p["symbol"],
                            "quantity": p["quantity"],
                            "price": p["order_price"],
                            "market_value": "PENDING",
                            "status": f"Pending ({p['action']})"
                        })
            except Exception as pe:
                if self.verbose: print(f"Error merging pending orders: {pe}")

            return positions
        except Exception as e:
            if self.verbose: print(f"Error during open positions scrape: {e}")
            return []

    def get_pending_orders(self):
        self.ensure_driver()
        try:
            self.driver.get("https://app.howthemarketworks.com/trading/orderhistory?status=Open")
            wait = WebDriverWait(self.driver, 10)
            
            rows = []
            try:
                # Target the order history table rows
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".large-9.columns table tbody tr")))
                rows = self.driver.find_elements(By.CSS_SELECTOR, ".large-9.columns table tbody tr")
            except Exception as e:
                # If no pending orders, the table might have "No data" or just be empty
                return []

            pending = []
            for row in rows:
                cols = row.find_elements(By.TAG_NAME, "td")
                # Mapping: 0=Date, 1=Action, 2=Symbol, 3=Qty, 4=Order Price, 8=Order#, 9=Status
                if len(cols) >= 5:
                    symbol = cols[2].text.strip()
                    if symbol and symbol != "Symbol": # Avoid header if it exists
                        pending.append({
                            "symbol": symbol,
                            "quantity": cols[3].text.strip(),
                            "order_price": cols[4].text.strip(),
                            "action": cols[1].text.strip(),
                            "order_id": cols[8].text.strip() if len(cols) > 8 else "N/A"
                        })
            return pending
        except Exception as e:
            if self.verbose: print(f"Error during pending orders scrape: {e}")
            return []

    def trade(self, symbol, action, quantity, order_type="Market", limit_stop_price=0, order_term='Good for Day'):
        self.ensure_driver()
        if self.verbose: logger.info(f"Trading {action} {quantity} shares of {symbol} at {order_type}...")
        try:
            self.driver.get("https://app.howthemarketworks.com/trading/equities")
            wait = WebDriverWait(self.driver, 10)

            # Help the page load and remove potential blockers
            self.driver.execute_script("""
                // Remove ad-blocker notice and other overlays that might intercept clicks
                const blockers = document.querySelectorAll('#adblocker-notice, .qmod-ui-modal-backdrop, .cookie-banner');
                blockers.forEach(el => el.remove());
            """)

            # Select Action
            action_select = Select(wait.until(EC.presence_of_element_located((By.ID, "ddlOrderSide"))))
            action_select.select_by_visible_text(action)

            # Enter Symbol
            symbol_input = wait.until(EC.presence_of_element_located((By.ID, "tbSymbol")))
            symbol_input.clear()
            symbol_input.send_keys(symbol)

            # Enter Quantity
            quantity_input = wait.until(EC.presence_of_element_located((By.ID, "tbQuantity")))
            quantity_input.clear()
            quantity_input.send_keys(str(quantity))

            # Select Order Type
            type_select = Select(wait.until(EC.presence_of_element_located((By.ID, "ddlOrderType"))))
            type_select.select_by_visible_text(order_type)

            if order_type != "Market":
                limit_stop_input = wait.until(EC.presence_of_element_located((By.ID, "tbPrice")))
                limit_stop_input.clear()
                limit_stop_input.send_keys(str(limit_stop_price))

                order_term_select = Select(wait.until(EC.presence_of_element_located((By.ID, "ddlOrderExpiration"))))
                order_term_select.select_by_visible_text(order_term)

            # Wait for button to be VISIBLE and click via JS
            preview_btn = wait.until(EC.visibility_of_element_located((By.ID, "btn-preview-order")))
            self.driver.execute_script("arguments[0].click();", preview_btn)

            # Wait for Confirm Button to be VISIBLE and Click via JS
            confirm_btn = wait.until(EC.visibility_of_element_located((By.ID, "btn-place-order")))
            # Small delay to ensure any animation finishes
            time.sleep(1) 
            self.driver.execute_script("arguments[0].click();", confirm_btn)
            
            return f"Order to {action} {quantity} of {symbol} submitted successfully."
        except Exception as e:
            return f"Error during trading: {e}"

    def get_ticker_details(self, symbol):
        self.ensure_driver()
        try:
            url = f"https://app.howthemarketworks.com/quotes/quotes?type=detailedquotetabchartnews&symbol={symbol}"
            self.driver.get(url)
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "span.qmod-last")))

            data = {}
            selectors = {
                "price": "span.qmod-last",
                "day_change": "span.qmod-chg-total",
                "day_change_pct": "span.qmod-chg-percent",
                "volume": ".qmod-quote-element-sharevolume .qmod-data-point",
                "market_cap": ".qmod-quote-element-marketcap .qmod-data-point",
                "pe_ratio": ".qmod-quote-element-peratio .qmod-data-point",
                "week_52_high": ".qmod-quote-element-week52high span.qmod-data-point",
                "week_52_low": ".qmod-quote-element-week52low span.qmod-data-point",
                "dividend": ".qmod-quote-element-dividend .qmod-data-point"
            }

            for key, css in selectors.items():
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, css)
                    data[key] = element.text.strip()
                except:
                    data[key] = "N/A"
            return data
        except Exception as e:
            return {"error": str(e)}

    def get_ticker_news(self, symbol, quantity=5):
        self.ensure_driver()
        try:
            url = f"https://app.howthemarketworks.com/quotes/quotes?type=fullnewssummary&symbol={symbol}"
            self.driver.get(url)
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li.qmod-newsitem")))

            news_items = []
            elements = self.driver.find_elements(By.CSS_SELECTOR, "li.qmod-newsitem")

            for el in elements[:quantity]:
                try:
                    headline = el.find_element(By.CSS_SELECTOR, "a.qmod-headline").text.strip()
                    date = el.find_element(By.CSS_SELECTOR, "span.qmod-news-date").text.strip()
                    storyid = el.find_element(By.CSS_SELECTOR, "a.qmod-headline").get_attribute("data-storyid")
                    news_items.append({"symbol": symbol, "headline": headline, "date": date, "storyid": storyid})
                except Exception:
                    continue
            return news_items
        except Exception as e:
            return []

    def get_article_by_storyid(self, symbol: str, storyid: str):
        self.ensure_driver()
        try:
            url = f"https://app.howthemarketworks.com/quotes/quotes?type=fullnewssummary&symbol={symbol}"
            self.driver.get(url)
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li.qmod-newsitem")))

            headline_selector = f"a.qmod-headline[data-storyid='{storyid}']"
            headline_el = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, headline_selector)))
            self.driver.execute_script("arguments[0].click();", headline_el)

            wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".qmod-ui-modal")))
            
            article_link = None
            try:
                modal_body = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".qmod-ui-modal-body")))
                try:
                    story_div = modal_body.find_element(By.CSS_SELECTOR, ".qmod-story")
                    article_link = story_div.find_element(By.TAG_NAME, "a").get_attribute("href")
                except:
                    pass
                
                if not article_link:
                    links = modal_body.find_elements(By.TAG_NAME, "a")
                    for link in links:
                        href = link.get_attribute("href")
                        if href and "http" in href and "quotemedia" not in href and "javascript" not in href:
                            article_link = href
                            break
            except Exception as e:
                pass

            try:
                close_btn = self.driver.find_element(By.CSS_SELECTOR, ".qmod-ui-modal-close, .qmod-close-modal")
                close_btn.click()
                time.sleep(0.5)
            except:
                pass
            return article_link
        except Exception as e:
            return None

    def get_portfolio_summary(self):
        self.ensure_driver()
        try:
            self.driver.get("https://app.howthemarketworks.com/accounting/dashboard")
            wait = WebDriverWait(self.driver, 10)
            
            # Use specific, verified IDs for absolute reliability
            try:
                # Wait for at least one indicator that the dashboard is loaded
                wait.until(EC.presence_of_element_located((By.ID, "portfolioValue")))
                
                import re
                def get_clean_value(el_id):
                    try:
                        text = self.driver.find_element(By.ID, el_id).text
                        # HTMW summary items often contain labels, icons, and then the value.
                        # Using regex to find the currency pattern (e.g. $100,000.00) is most robust.
                        match = re.search(r"\$[0-9,.]+", text)
                        return match.group(0) if match else text.strip()
                    except: return "N/A"

                return {
                    "net_worth": get_clean_value("portfolioValue"),
                    "cash": get_clean_value("cashBalance"),
                    "buying_power": get_clean_value("buyingPower")
                }
            except Exception as e:
                if self.verbose: print(f"Dashboard summary IDs not found, trying fallback: {e}")
                summary_el = self.driver.find_element(By.CLASS_NAME, "summary")
                return {"info": summary_el.text.strip()}
        except Exception as e:
            if self.verbose: print(f"Error in get_portfolio_summary: {e}")
            return {"error": str(e)}

    def get_market_movers(self):
        self.ensure_driver()
        movers = {}
        try:
            url = "https://app.howthemarketworks.com/quotes/quotes?type=marketmovers"
            self.driver.get(url)
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".qmod-marketmovers-table-title")))

            panels = self.driver.find_elements(By.CSS_SELECTOR, ".qmod-panel")
            for panel in panels:
                try:
                    title_el = panel.find_element(By.CSS_SELECTOR, ".qmod-marketmovers-table-title")
                    title = title_el.text.strip()
                    symbols = []
                    rows = panel.find_elements(By.CSS_SELECTOR, "table.qmod-marketmovers-table tbody tr")
                    for row in rows:
                        try:
                            sym_el = row.find_element(By.CSS_SELECTOR, "td.qmod-col-symbol")
                            symbols.append(sym_el.text.strip())
                        except:
                            continue
                    if title and symbols:
                        movers[title] = symbols
                except:
                    continue
            return movers
        except Exception as e:
            return {}

    def get_analyst_ratings(self, symbol):
        self.ensure_driver()
        try:
            url = f"https://app.howthemarketworks.com/quotes/quotes?type=analyst&symbol={symbol}"
            self.driver.get(url)
            wait = WebDriverWait(self.driver, 10)
            # Wait for either the tool container or a "no data" message
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".qmod-analyst-ar, .qmod-analyst-crd, .qmod-analyst")))
            except:
                # Check if it's just a "No data available" message
                if "no data" in self.driver.page_source.lower():
                    return {"recommendation": "None", "mean_score": "N/A", "rating_counts": {}, "info": "No analyst coverage found for this symbol."}
                raise

            data = {}
            try:
                # Summary rating text (e.g. "Moderate Buy")
                summary_el = self.driver.find_element(By.CSS_SELECTOR, "div.qmod-rating, .qmod-reco-summary")
                data['recommendation'] = summary_el.text.strip()
            except: data['recommendation'] = "N/A"

            try:
                # Mean score (often a number like 2.00)
                mean_el = None
                try:
                    mean_el = self.driver.find_element(By.CSS_SELECTOR, ".qmod-analyst-crd .pure-u-1-4 div")
                except: pass
                
                if mean_el and any(char.isdigit() for char in mean_el.text):
                    data['mean_score'] = mean_el.text.strip()
                else:
                    # Fallback strategy: find by text content
                    try:
                        mean_label = self.driver.find_element(By.XPATH, "//*[contains(text(), 'Current Average Recommendation')]/following-sibling::*")
                        data['mean_score'] = mean_label.text.strip()
                    except:
                        data['mean_score'] = "N/A"
            except: data['mean_score'] = "N/A"

            # Parse counts table
            counts = {}
            try:
                # Look for the table within the Analyst Recommendations section
                rows = self.driver.find_elements(By.CSS_SELECTOR, ".qmod-analyst-ar table.qmod-table tbody tr, .qmod-analyst-reco-table tbody tr")
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 2:
                        label = cells[0].text.strip()
                        value = cells[1].text.strip()
                        if label and value:
                            counts[label] = value
            except: pass
            data['rating_counts'] = counts
            
            return data
        except Exception as e:
            return {"error": str(e)}

    def get_price_history(self, symbol):
        self.ensure_driver()
        try:
            url = f"https://app.howthemarketworks.com/quotes/quotes?type=pricehistorydownload1&symbol={symbol}"
            self.driver.get(url)
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".qmod-table")))

            history = []
            rows = self.driver.find_elements(By.CSS_SELECTOR, ".qmod-table tbody tr")
            for row in rows[:20]: # Limit to last 20 entries
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 8:
                    # Detect if columns are shifted (e.g. first cell is empty)
                    idx_offset = 0
                    if not cells[0].text.strip() and cells[1].text.strip():
                        idx_offset = 1
                    
                    history.append({
                        "date": cells[0 + idx_offset].text.strip(),
                        "open": cells[1 + idx_offset].text.strip(),
                        "high": cells[2 + idx_offset].text.strip(),
                        "low": cells[3 + idx_offset].text.strip(),
                        "close": cells[4 + idx_offset].text.strip(),
                        "vwap": cells[5 + idx_offset].text.strip(),
                        "volume": cells[6 + idx_offset].text.strip(),
                        "change_pct": cells[7 + idx_offset].text.strip()
                    })
            return history
        except Exception as e:
            return {"error": str(e)}

    def close(self):
        self.driver.quit()

# MCP Server Implementation
mcp = FastMCP("HTMW Trader", log_level="ERROR")

# Global trader instance
_trader = None

def get_trader():
    global _trader
    if _trader is None:
        load_dotenv()
        username = os.getenv("HTMW_USERNAME") or os.getenv("USERNAME")
        password = os.getenv("HTMW_PASSWORD") or os.getenv("PASSWORD")
        if not username or not password:
            raise ValueError("HTMW_USERNAME and HTMW_PASSWORD environment variables must be set")
        _trader = HTMWTrader(username, password, headless=True)
    return _trader

@mcp.tool()
def get_portfolio_summary():
    """Get the current portfolio summary including net worth, cash, and buying power."""
    return get_trader().get_portfolio_summary()

@mcp.tool()
def get_open_positions():
    """Get a list of all currently open stock positions, including pending orders."""
    return get_trader().get_open_positions()

@mcp.tool()
def get_pending_orders():
    """Get a list of all active orders that have not yet been filled."""
    return get_trader().get_pending_orders()

@mcp.tool()
def trade_stock(symbol: str, action: str, quantity: int, order_type: str = "Market", limit_stop_price: float = 0, order_term: str = 'Good for Day'):
    """
    Execute a stock trade.
    - action: 'Buy', 'Sell', 'Short', 'Cover'
    - order_type: 'Market', 'Limit', 'Stop', 'Trailing Stop'
    - order_term: 'Good for Day', 'Good til Cancel', 'Good til Date'
    """
    return get_trader().trade(symbol, action, quantity, order_type, limit_stop_price, order_term)

@mcp.tool()
def get_ticker_details(symbol: str):
    """Get detailed financial information for a specific stock symbol."""
    return get_trader().get_ticker_details(symbol)

@mcp.tool()
def get_ticker_news(symbol: str, quantity: int = 5):
    """Get recent news headlines and story IDs for a specific stock symbol."""
    return get_trader().get_ticker_news(symbol, quantity)

@mcp.tool()
def get_article_url(symbol: str, storyid: str):
    """Get the full article URL for a news story given its symbol and storyid."""
    return get_trader().get_article_by_storyid(symbol, storyid)

@mcp.tool()
def get_market_movers():
    """Get lists of market movers including top gainers, losers, and most active stocks."""
    return get_trader().get_market_movers()

@mcp.tool()
def get_analyst_ratings(symbol: str):
    """Get consensus analyst recommendations and rating counts for a stock."""
    return get_trader().get_analyst_ratings(symbol)

@mcp.tool()
def get_price_history(symbol: str):
    """Get recent historical price data (Date, Open, High, Low, Close, Volume, etc.) for a stock."""
    return get_trader().get_price_history(symbol)


if __name__ == "__main__":
    mcp.run()
