import asyncio
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ResearchAgent:
    async def run(self, query: str):
        async with AsyncExitStack() as stack:
            # Tell MCP how to start the server
            server_params = StdioServerParameters(
                command="python3",
                args=["server.py"],
                env=None,
            )

            # Create stdio transport
            stdio_transport = await stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport

            # Create MCP session
            session = await stack.enter_async_context(
                ClientSession(stdio, write)
            )

            # Initialize MCP session
            await session.initialize()

            # Call MCP tool
            result = await session.call_tool(
                "search_web",
                {"query": query},
            )

            return result
