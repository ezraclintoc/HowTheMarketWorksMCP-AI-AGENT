import sys
import asyncio
import json
import os
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QTableWidget, 
                             QTableWidgetItem, QTextEdit, QFrame, QHeaderView,
                             QScrollArea, QSplitter)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QObject, QThread
from PySide6.QtGui import QFont, QColor, QPalette, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtWidgets import QPushButton
import signal

import trading_engine

# EngineWorker is retired in favor of direct polling for better stability

class AsyncThread(QThread):
    def __init__(self, coro):
        super().__init__()
        self.coro = coro

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.coro)

class ModernFrame(QFrame):
    def __init__(self, title, icon=None, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            ModernFrame {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 6px;
            }
            QLabel#title {
                color: #58a6ff;
                font-weight: 800;
                font-size: 11px;
                letter-spacing: 0.5px;
                padding: 4px;
            }
        """)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 8, 10, 10)
        
        self.title_layout = QHBoxLayout()
        self.title_label = QLabel(title.upper())
        self.title_label.setObjectName("title")
        self.title_layout.addWidget(self.title_label)
        self.title_layout.addStretch()
        
        if icon:
            icon_lbl = QLabel(icon)
            icon_lbl.setStyleSheet("color: #30363d; font-size: 14px;")
            self.title_layout.addWidget(icon_lbl)
            
        self.layout.addLayout(self.title_layout)

class PortfolioHeader(ModernFrame):
    def __init__(self, parent=None):
        super().__init__("Market Overview", "📊", parent)
        self.setFixedHeight(110)
        grid = QHBoxLayout()
        grid.setSpacing(40)
        
        self.net_worth_lbl = QLabel("$0.00")
        self.cash_lbl = QLabel("$0.00")
        self.buying_power_lbl = QLabel("$0.00")
        self.time_lbl = QLabel("00:00:00")
        
        def style_val(lbl, color="#58a6ff"):
            lbl.setStyleSheet(f"font-size: 26px; font-weight: 900; color: {color}; font-family: 'Consolas', monospace;")
            
        style_val(self.net_worth_lbl, "#3fb950") # Success Green
        style_val(self.cash_lbl, "#58a6ff")      # Info Blue
        style_val(self.buying_power_lbl, "#bc8cff") # Mystery Purple
        style_val(self.time_lbl, "#8b949e")
        self.time_lbl.setStyleSheet("font-size: 20px; font-weight: bold; color: #30363d;")
        
        def create_block(label, val_lbl):
            container = QVBoxLayout()
            l = QLabel(label)
            l.setStyleSheet("font-size: 10px; color: #8b949e; font-weight: bold;")
            container.addWidget(l)
            container.addWidget(val_lbl)
            return container

        grid.addLayout(create_block("PORTFOLIO VALUE", self.net_worth_lbl))
        grid.addLayout(create_block("CASH RESERVE", self.cash_lbl))
        grid.addLayout(create_block("TRADING POWER", self.buying_power_lbl))
        grid.addStretch()
        grid.addLayout(create_block("SYSTEM TIME", self.time_lbl))
        
        self.layout.addLayout(grid)
        
        # Timer for clock
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

    def update_clock(self):
        self.time_lbl.setText(datetime.now().strftime("%H:%M:%S"))

    def update_snapshot(self, data):
        self.net_worth_lbl.setText(str(data.get("net_worth", "—")))
        self.cash_lbl.setText(str(data.get("cash", "—")))
        self.buying_power_lbl.setText(str(data.get("buying_power", "—")))

class TickerDetailOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setStyleSheet("background-color: rgba(13, 17, 23, 0.95); border: 1px solid #30363d; border-radius: 10px;")
        self.hide()
        
        layout = QVBoxLayout(self)
        self.title = QLabel("TICKER DETAILS")
        self.title.setStyleSheet("font-size: 18px; color: #f0f6fc; font-weight: bold;")
        layout.addWidget(self.title)
        
        self.info = QLabel("Loading details...")
        self.info.setStyleSheet("color: #8b949e;")
        layout.addWidget(self.info)
        
        close_btn = QLabel("Click to close")
        close_btn.setStyleSheet("color: #58a6ff; font-size: 10px;")
        layout.addWidget(close_btn)

    def mousePressEvent(self, event):
        self.hide()

class TradingGUI(QMainWindow):
    def __init__(self, provider=None):
        super().__init__()
        self.provider = provider
        self.setWindowTitle("HTMW Pro Trading Station")
        self.resize(1400, 900)
        self.setup_ui()
        self.setup_engine()

    def setup_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #010409; }
            QLabel { color: #c9d1d9; font-family: 'Inter', 'Segoe UI', sans-serif; }
            QTableWidget {
                background-color: #0d1117;
                color: #c9d1d9;
                gridline-color: #21262d;
                border: none;
                font-family: 'Inter', sans-serif;
                selection-background-color: #161b22;
            }
            QHeaderView::section {
                background-color: #010409;
                color: #8b949e;
                padding: 6px;
                border: none;
                border-bottom: 2px solid #21262d;
                font-weight: bold;
                font-size: 10px;
            }
            QScrollBar:vertical {
                border: none;
                background: #0d1117;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #30363d;
                min-height: 20px;
                border-radius: 4px;
            }
            QTextEdit {
                background-color: #0d1117;
                color: #c9d1d9;
                border: none;
                font-family: 'Fira Code', 'Consolas', monospace;
                font-size: 12px;
                line-height: 1.4;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Header
        self.header = PortfolioHeader()
        main_layout.addWidget(self.header)

        # Middle Section (Splitter)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left Column
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        
        self.positions_table = QTableWidget(0, 3)
        self.positions_table.setHorizontalHeaderLabels(["SYMBOL", "VALUATION", "UNREALIZED P/L"])
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        pos_frame = ModernFrame("Portfolio Holdings", "💼")
        pos_frame.layout.addWidget(self.positions_table)
        left_layout.addWidget(pos_frame, 3)
        
        self.perf_container = QWidget()
        perf_vbox = QVBoxLayout(self.perf_container)
        self.performance_stats = QLabel("Calculating...")
        self.performance_stats.setStyleSheet("font-size: 13px; color: #8b949e;")
        perf_vbox.addWidget(self.performance_stats)
        
        perf_frame = ModernFrame("Analytics Engine", "📈")
        perf_frame.layout.addWidget(self.perf_container)
        left_layout.addWidget(perf_frame, 1)

        # Center Column
        center_col = QWidget()
        center_layout = QVBoxLayout(center_col)
        
        # Chart
        self.figure = Figure(figsize=(5, 3), dpi=90, facecolor='#0d1117')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#0d1117')
        self.ax.tick_params(colors='#30363d', labelsize=8)
        for spine in self.ax.spines.values(): spine.set_color('#21262d')
        
        chart_frame = ModernFrame("Equity Growth Curve", "📉")
        chart_frame.layout.addWidget(self.canvas)
        center_layout.addWidget(chart_frame, 2)
        
        # Activity Feed
        self.activity_log = QTextEdit()
        self.activity_log.setReadOnly(True)
        feed_frame = ModernFrame("Real-time Intelligence Feed", "📡")
        
        self.progress_bar = QLabel("Next cycle in 15:00")
        self.progress_bar.setStyleSheet("font-size: 9px; color: #30363d; font-weight: bold;")
        feed_frame.title_layout.insertWidget(1, self.progress_bar)
        
        feed_frame.layout.addWidget(self.activity_log)
        center_layout.addWidget(feed_frame, 3)

        # Right Column
        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)
        
        self.watchlist_table = QTableWidget(0, 1)
        self.watchlist_table.setHorizontalHeaderLabels(["TICKER"])
        self.watchlist_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        watch_frame = ModernFrame("Global Watchlist", "👁️")
        watch_frame.layout.addWidget(self.watchlist_table)
        right_layout.addWidget(watch_frame, 3)
        
        self.status_box = QWidget()
        status_l = QVBoxLayout(self.status_box)
        self.settings_info = QLabel("STRATEGY: DIVERSIFIED\nRISK: MODERATE")
        self.settings_info.setStyleSheet("font-size: 11px; line-height: 1.6; color: #58a6ff; font-weight: bold;")
        status_l.addWidget(self.settings_info)
        
        self.pulse = QLabel(" ● SYSTEM ONLINE")
        self.pulse.setStyleSheet("color: #3fb950; font-weight: 800; font-size: 10px;")
        status_l.addWidget(self.pulse)
        
        self.refresh_btn = QPushButton("🔄 REFRESH NOW")
        self.refresh_btn.setCursor(Qt.PointingHandCursor)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 6px;
                font-size: 10px;
                font-weight: bold;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #30363d;
                border: 1px solid #8b949e;
            }
            QPushButton:pressed {
                background-color: #161b22;
            }
        """)
        self.refresh_btn.clicked.connect(trading_engine.force_cycle)
        status_l.addWidget(self.refresh_btn)
        
        settings_frame = ModernFrame("Engine Status", "⚙️")
        settings_frame.layout.addWidget(self.status_box)
        right_layout.addWidget(settings_frame, 1)

        splitter.addWidget(left_col)
        splitter.addWidget(center_col)
        splitter.addWidget(right_col)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
        
        # Overlay
        self.overlay = TickerDetailOverlay(self)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()

    def closeEvent(self, event):
        """Ensure threads stop when window is closed."""
        self.engine_thread.terminate()
        event.accept()

    def setup_engine(self):
        self.event_queue = trading_engine.events.subscribe()
        
        # Start engine in background thread
        self.engine_thread = AsyncThread(trading_engine.run_trader(auto_mode=True, provider=self.provider))
        self.engine_thread.start()
        
        # Poll for events in the main thread using a QTimer
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_events)
        self.poll_timer.start(100) # Poll every 100ms

    def poll_events(self):
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                self.handle_event(event)
            except:
                break

    @Slot(dict)
    def handle_event(self, event):
        etype = event["type"]
        data = event["data"]
        
        if etype == "ai_log":
            tag = "INTEL"
            color = "#58a6ff"
            if "[Strategy]" in data or "decided" in data.lower():
                tag, color = "STRAT", "#bc8cff"
            elif "[Analyst" in data:
                tag, color = "ALFA", "#ffa657"
            elif "Trade" in data:
                tag, color = "EXEC", "#3fb950"
                
            self.activity_log.append(f"<span style='color:#30363d'>[{datetime.now().strftime('%H:%M:%S')}]</span> "
                                    f"<span style='color:{color}; font-weight:bold;'>[{tag}]</span> {data}")
        elif etype == "system_log":
            self.activity_log.append(f"<span style='color:#30363d'>[{datetime.now().strftime('%H:%M:%S')}]</span> "
                                    f"<span style='color:#8b949e; font-style:italic;'>{data}</span>")
        elif etype == "portfolio_snapshot":
            self.header.update_snapshot(data)
        elif etype == "positions":
            self.update_positions(data)
        elif etype == "watchlist":
            self.update_watchlist(data)
        elif etype == "portfolio_value":
            self.update_chart(data)
        elif etype == "performance":
            self.update_performance(data)
        elif etype == "status":
            self.statusBar().showMessage(f"Engine Status: {data}")
        elif etype == "cycle_timer":
            mins, secs = divmod(data, 60)
            self.progress_bar.setText(f"NEXT CYCLE IN {mins:02d}:{secs:02d}")
        elif etype == "heartbeat":
            self.trigger_pulse()

    def trigger_pulse(self):
        self.pulse.setStyleSheet("color: #3fb950; font-weight: 800; font-size: 10px; text-shadow: 0 0 5px #3fb950;")
        QTimer.singleShot(500, lambda: self.pulse.setStyleSheet("color: #3fb950; font-weight: 800; font-size: 10px;"))

    def update_positions(self, data):
        self.positions_table.setRowCount(len(data))
        for i, pos in enumerate(data):
            sym_item = QTableWidgetItem(pos.get("symbol", "—").split('\n')[0].strip())
            sym_item.setFont(QFont("Inter", 10, QFont.Bold))
            self.positions_table.setItem(i, 0, sym_item)
            
            val_item = QTableWidgetItem(str(pos.get("market_value", "—")))
            val_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.positions_table.setItem(i, 1, val_item)
            
            pl = str(pos.get("total_gain_loss", "—"))
            pl_item = QTableWidgetItem(pl)
            pl_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if "+" in pl: pl_item.setForeground(QColor("#3fb950"))
            elif "-" in pl: pl_item.setForeground(QColor("#f85149"))
            self.positions_table.setItem(i, 2, pl_item)

    def update_watchlist(self, data):
        self.watchlist_table.setRowCount(len(data))
        for i, sym in enumerate(data):
            item = QTableWidgetItem(str(sym).upper())
            item.setTextAlignment(Qt.AlignCenter)
            self.watchlist_table.setItem(i, 0, item)

    def update_chart(self, values):
        self.ax.clear()
        if len(values) > 1:
            x = np.arange(len(values))
            self.ax.plot(x, values, color='#3fb950', linewidth=2.5, antialiased=True)
            self.ax.fill_between(x, values, min(values)*0.98, color='#3fb950', alpha=0.1)
        else:
            self.ax.plot(values, color='#3fb950', linewidth=2, marker='o')
            
        self.ax.set_facecolor('#0d1117')
        self.ax.grid(True, color='#21262d', linestyle='-', alpha=0.3)
        self.ax.set_xticks([])
        self.canvas.draw()

    def update_performance(self, data):
        txt = (f"<b>ROI:</b> {data.get('roi')}%<br>"
               f"<b>Win Rate:</b> {data.get('win_rate')}%<br>"
               f"<b>Trades:</b> {data.get('total_trades')}")
        self.performance_stats.setText(txt)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TradingGUI()
    window.show()
    sys.exit(app.exec())
