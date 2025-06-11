from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv("../.env")

#create a MCP server
mcp = FastMCP(
    name="Calculator",
    host="0.0.0.0", #only used for SSE transport (localhost)
    port=8050, # only used for SSE transport (set this to any port)
)

#add a simple calculator tool
@mcp.tool() #the decorator registers the function as a tool and mcp is defined above
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

#run the server
if __name__ == "__main__":
    transport = 'sse'
    if transport == "stdio":
        print("Running server with stdio ransport")
        mcp.run(transport="stdio")
    elif transport == "sse":
        print("Running server with SSE transport")
        mcp.run(transport="sse")
    else:
        raise ValueError(f"Unknow transport : {transport}. Use 'stdio' or 'sse'.")
        

