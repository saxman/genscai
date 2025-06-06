from mcp.server.fastmcp import FastMCP

server = FastMCP("Local Agent Helper")


@server.tool()
def ls(directory: str) -> str:
    "List the contents of a directory."
    import os

    return "\n".join(os.listdir(directory))


@server.tool()
def cat(file: str) -> str:
    "Read the contents of a file."
    try:
        with open(file, "r") as f:
            return f.read()
    except:
        return ""


@server.tool()
def echo(message: str, file: str) -> str:
    "Write text to a file."
    try:
        with open(file, "w") as f:
            f.write(message)
            return "success"
    except:
        return "failed"
