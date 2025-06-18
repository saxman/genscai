from genscai.tools import MCPClient

import nest_asyncio
nest_asyncio.apply()

mcp_config = {
    "mcpServers": {
        "genscai": {"command": "python", "args": ["genscai/tools.py"]}
    }
}

# mcp_client = MCPClient(mcp_config)
mcp_client = MCPClient()

tools = mcp_client.list_tools()
for tool in tools:
    print(f"Tool Name: {tool.name}")    
    print(f"Description: {tool.description}")
    print(f"Input schema: {tool.inputSchema}")
    print("-" * 40)

response = mcp_client.call_tool(
    tool_name="hello",
    params={"name": "World"}
)

print(response)

response = mcp_client.call_tool(
    tool_name="search_research_articles",
    params={"search_request": "malaria modeling"}
)

print(response)
