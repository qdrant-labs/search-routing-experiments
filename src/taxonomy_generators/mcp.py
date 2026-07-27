from mcp.server.fastmcp import FastMCP
from taxonomy_generators.tools import build_tools

def main() -> None:

    server = FastMCP("taxonomy-generators")
    for spec in build_tools():
        server.add_tool(spec.run, name=spec.name, description=spec.description)
    server.run()


if __name__ == "__main__":
    main()
