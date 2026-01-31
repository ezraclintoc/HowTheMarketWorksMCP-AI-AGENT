import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os

async def test_tools():
    # Use the venv python to run the server
    server_params = StdioServerParameters(
        command="./.venv/bin/python",
        args=["mcp_scrape.py"],
        env=os.environ.copy()
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("\n--- Testing get_portfolio_summary ---")
            summary = await session.call_tool("get_portfolio_summary", {})
            print(summary.content[0].text)
            
            print("\n--- Testing get_open_positions ---")
            positions = await session.call_tool("get_open_positions", {})
            print(positions.content[0].text)
            
            print("\n--- Testing get_ticker_details (AAPL) ---")
            details = await session.call_tool("get_ticker_details", {"symbol": "AAPL"})
            print(details.content[0].text)
            
            print("\n--- Testing get_analyst_ratings (AAPL) ---")
            ratings = await session.call_tool("get_analyst_ratings", {"symbol": "AAPL"})
            print(ratings.content[0].text)

if __name__ == "__main__":
    asyncio.run(test_tools())
