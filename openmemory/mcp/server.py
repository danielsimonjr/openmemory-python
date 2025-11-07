"""
MCP Server for OpenMemory

Exposes memory operations via Model Context Protocol.
"""
import asyncio
import json
import sys
from typing import Any, Dict
from mcp.server.models import InitializationOptions
from mcp.server import Server
from mcp.types import Tool, TextContent

from ..memory_system import MemorySystem


# Initialize memory system
memory_system = None


def get_memory_system() -> MemorySystem:
    """Get or create memory system"""
    global memory_system
    if memory_system is None:
        memory_system = MemorySystem()
    return memory_system


# Create MCP server
server = Server("openmemory")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="add_memory",
            description="Add a new memory to the system",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Memory content"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional user ID for isolation"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata"
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="query_memory",
            description="Query memories using semantic search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query string"
                    },
                    "k": {
                        "type": "number",
                        "description": "Number of results (default: 10)"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional user ID filter"
                    },
                    "sectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional sector filter"
                    },
                    "min_salience": {
                        "type": "number",
                        "description": "Minimum salience threshold"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="reinforce_memory",
            description="Manually reinforce a memory to increase its salience",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Memory ID to reinforce"
                    },
                    "boost": {
                        "type": "number",
                        "description": "Salience boost amount (default: 0.1)"
                    }
                },
                "required": ["memory_id"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> list[TextContent]:
    """Handle tool calls"""
    mem_sys = get_memory_system()

    try:
        if name == "add_memory":
            result = await mem_sys.add_memory(
                content=arguments["content"],
                user_id=arguments.get("user_id"),
                tags=arguments.get("tags"),
                metadata=arguments.get("metadata")
            )
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        elif name == "query_memory":
            results = await mem_sys.query(
                query=arguments["query"],
                k=arguments.get("k", 10),
                user_id=arguments.get("user_id"),
                sectors=arguments.get("sectors"),
                min_salience=arguments.get("min_salience")
            )

            # Convert results to dict
            results_dict = [
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    "primary_sector": r.primary_sector,
                    "sectors": r.sectors,
                    "salience": r.salience
                }
                for r in results
            ]

            return [TextContent(
                type="text",
                text=json.dumps(results_dict, indent=2)
            )]

        elif name == "reinforce_memory":
            await mem_sys.reinforce_memory(
                memory_id=arguments["memory_id"],
                boost=arguments.get("boost", 0.1)
            )
            return [TextContent(
                type="text",
                text=json.dumps({"status": "success", "memory_id": arguments["memory_id"]})
            )]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]


async def main():
    """Main entry point for MCP server"""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="openmemory",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
