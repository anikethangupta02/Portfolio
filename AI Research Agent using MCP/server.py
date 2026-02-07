from mcp.server.fastmcp import FastMCP
import requests

mcp= FastMCP("Researcher Agent Tool")

@mcp.tool()
def search_web(query: str) -> str:
    """Web Search Tool"""
    return f"Search results for: {query}"

@mcp.tool()
def summarize(text: str) -> str:
    """Summarizer Tool"""
    return text[:300] + "..."

if __name__=="__main__":
    mcp.run()