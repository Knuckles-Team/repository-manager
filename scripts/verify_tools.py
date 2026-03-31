
import asyncio
import os
import json
import logging
from pydantic_ai.mcp import load_mcp_servers
from agent_utilities.mcp_utilities import get_mcp_config_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    mcp_config = get_mcp_config_path()
    print(f"Loading MCP config from: {mcp_config}")

    if os.path.exists(mcp_config):
        try:
            with open(mcp_config, "r") as f:
                config_data = json.load(f)


            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_config:
                json.dump(config_data, temp_config)
                temp_config_path = temp_config.name

            print(f"Temp config created at: {temp_config_path}")

            try:









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
