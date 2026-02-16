
import asyncio
import os
import json
import logging
from pydantic_ai.mcp import load_mcp_servers
from repository_manager.utils import get_mcp_config_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    mcp_config = get_mcp_config_path()
    print(f"Loading MCP config from: {mcp_config}")

    if os.path.exists(mcp_config):
        try:
            with open(mcp_config, "r") as f:
                config_data = json.load(f)

            # Create temp file as in agent code
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_config:
                json.dump(config_data, temp_config)
                temp_config_path = temp_config.name

            print(f"Temp config created at: {temp_config_path}")

            try:
                # Load tools
                # Note: load_mcp_servers is synchronous but might use asyncio internally?
                # It returns a list of Tool objects.
                # Actually, looking at pydantic_ai docs/code, load_mcp_servers returns a list of tools.
                # But it might need an event loop running? It usually starts a client.
                # Let's see if we need to run it in a context manager?
                # The agent code calls `load_mcp_servers(temp_config_path)`.

                # Check pydantic_ai version/implementation.
                # Assuming it works as in agent code.
                tools = load_mcp_servers(temp_config_path)
                print(f"Loaded {len(tools)} tools:")
                for tool in tools:
                    print(f" - {tool.name}: {tool.description}")

            except Exception as e:
                print(f"Error loading tools: {e}")
            finally:
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)

        except Exception as e:
            print(f"Error reading config: {e}")

if __name__ == "__main__":
    asyncio.run(main())
