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

import trading_engine

class EngineWorker(QObject):
    finished = Signal()
    event_received = Signal(dict)

    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self._running = True

    async def run_loop(self):
        while self._running:
            try:
                event = await self.queue.get()
                self.event_received.emit(event)
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker Error: {e}")
        self.finished.emit()

class AsyncThread(QThread):
    def __init__(self, coro):
        super().__init__()
        self.coro = coro

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.coro)

class ModernFrame(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            ModernFrame {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
            }
            QLabel#title {
                color: #8b949e;
                font-weight: bold;
                font-size: 13px;
                padding-bottom: 5px;
            }
        """)
        self.layout = QVBoxLayout(self)
        self.title_label = QLabel(title.upper())
        self.title_label.setObjectName("title")
        self.layout.addWidget(self.title_label)

class PortfolioHeader(ModernFrame):
    def __init__(self, parent=None):
        super().__init__("Portfolio Summary", parent)
        self.setFixedHeight(100)
        grid = QGridLayout()
        
        self.net_worth_lbl = QLabel("—")
        self.cash_lbl = QLabel("—")
        self.buying_power_lbl = QLabel("—")
        
        for lbl in [self.net_worth_lbl, self.cash_lbl, self.buying_power_lbl]:
            lbl.setStyleSheet("font-size: 24px; font-weight: bold; color: #58a6ff;")
        
        grid.addWidget(QLabel("NET WORTH"), 0, 0)
        grid.addWidget(self.net_worth_lbl, 1, 0)
        grid.addWidget(QLabel("CASH"), 0, 1)
        grid.addWidget(self.cash_lbl, 1, 1)
        grid.addWidget(QLabel("BUYING POWER"), 0, 2)
        grid.addWidget(self.buying_power_lbl, 1, 2)
        
        self.layout.addLayout(grid)

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
            QMainWindow { background-color: #0d1117; }
            QLabel { color: #c9d1d9; font-family: 'Segoe UI', Arial; }
            QTableWidget {
                background-color: #0d1117;
                color: #c9d1d9;
                gridline-color: #30363d;
                border: none;
            }
            QHeaderView::section {
                background-color: #161b22;
                color: #8b949e;
                padding: 4px;
                border: 1px solid #30363d;
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
        self.positions_table.setHorizontalHeaderLabels(["Symbol", "Value", "Profit/Loss"])
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        pos_frame = ModernFrame("Current Positions")
        pos_frame.layout.addWidget(self.positions_table)
        left_layout.addWidget(pos_frame, 2)
        
        self.performance_stats = QLabel("Gathering stats...")
        self.performance_stats.setStyleSheet("font-size: 14px; line-height: 1.5;")
        perf_frame = ModernFrame("Performance Analytics")
        perf_frame.layout.addWidget(self.performance_stats)
        left_layout.addWidget(perf_frame, 1)

        # Center Column
        center_col = QWidget()
        center_layout = QVBoxLayout(center_col)
        
        # Chart
        self.figure = Figure(figsize=(5, 4), dpi=100, facecolor='#161b22')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#0d1117')
        self.ax.tick_params(colors='#8b949e')
        for spine in self.ax.spines.values(): spine.set_color('#30363d')
        
        chart_frame = ModernFrame("Portfolio Valuation Trend")
        chart_frame.layout.addWidget(self.canvas)
        center_layout.addWidget(chart_frame, 1)
        
        # Activity Feed
        self.activity_log = QTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setStyleSheet("background-color: #0d1117; color: #8b949e; border: none; font-family: 'Consolas', monospace;")
        feed_frame = ModernFrame("Intelligence Activity Feed")
        feed_frame.layout.addWidget(self.activity_log)
        center_layout.addWidget(feed_frame, 2)

        # Right Column
        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)
        
        self.watchlist_table = QTableWidget(0, 1)
        self.watchlist_table.setHorizontalHeaderLabels(["Symbol"])
        self.watchlist_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        watch_frame = ModernFrame("Global Watchlist")
        watch_frame.layout.addWidget(self.watchlist_table)
        right_layout.addWidget(watch_frame, 2)
        
        self.settings_info = QLabel("Strategy: DIVERSIFIED\nRisk: MODERATE")
        settings_frame = ModernFrame("System Status")
        settings_frame.layout.addWidget(self.settings_info)
        right_layout.addWidget(settings_frame, 1)

        splitter.addWidget(left_col)
        splitter.addWidget(center_col)
        splitter.addWidget(right_col)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
        
        # Overlay
        self.overlay = TickerDetailOverlay(self)

    def setup_engine(self):
        self.event_queue = trading_engine.events.subscribe()
        self.worker = EngineWorker(self.event_queue)
        
        # Start engine in background thread
        self.engine_thread = AsyncThread(trading_engine.run_trader(auto_mode=True, provider=self.provider))
        self.engine_thread.start()
        
        # Start worker thread for GUI events
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(lambda: asyncio.run(self.worker.run_loop()))
        self.worker.event_received.connect(self.handle_event)
        self.worker_thread.start()

    @Slot(dict)
    def handle_event(self, event):
        etype = event["type"]
        data = event["data"]
        
        if etype == "ai_log":
            self.activity_log.append(f"<span style='color:#58a6ff'>[AI]</span> {data}")
        elif etype == "system_log":
            self.activity_log.append(f"<span style='color:#8b949e'>{data}</span>")
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

    def update_positions(self, data):
        self.positions_table.setRowCount(len(data))
        for i, pos in enumerate(data):
            self.positions_table.setItem(i, 0, QTableWidgetItem(pos.get("symbol", "—")))
            self.positions_table.setItem(i, 1, QTableWidgetItem(str(pos.get("market_value", "—"))))
            pl = str(pos.get("total_gain_loss", "—"))
            pl_item = QTableWidgetItem(pl)
            if "+" in pl: pl_item.setForeground(QColor("#3fb950"))
            elif "-" in pl: pl_item.setForeground(QColor("#f85149"))
            self.positions_table.setItem(i, 2, pl_item)

    def update_watchlist(self, data):
        self.watchlist_table.setRowCount(len(data))
        for i, sym in enumerate(data):
            self.watchlist_table.setItem(i, 0, QTableWidgetItem(str(sym).upper()))

    def update_chart(self, values):
        self.ax.clear()
        self.ax.plot(values, color='#3fb950', linewidth=2, marker='o', markersize=4)
        self.ax.set_facecolor('#0d1117')
        self.ax.grid(True, color='#30363d', linestyle='--', alpha=0.5)
        self.canvas.draw()

    def update_performance(self, data):
        txt = (f"<b>ROI:</b> {data.get('roi')}%<br>"
               f"<b>Win Rate:</b> {data.get('win_rate')}%<br>"
               f"<b>Trades:</b> {data.get('total_trades')}")
        self.performance_stats.setText(txt)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TradingGUI()
    window.show()
    sys.exit(app.exec())
