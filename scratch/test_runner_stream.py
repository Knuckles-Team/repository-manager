import asyncio
from unittest.mock import MagicMock
from agent_utilities.graph.runner import run_graph_stream


async def test_stream():
    graph = MagicMock()

    # Mock graph.run to emit an event
    async def mock_run(state, deps):
        from agent_utilities.graph.config_helpers import emit_graph_event

        emit_graph_event(deps.event_queue, "mock_event")
        await asyncio.sleep(0.1)

    graph.run = mock_run

    config = {"tag_prompts": {}, "mcp_toolsets": []}

    print("Starting stream...")
    async for line in run_graph_stream(graph, config, "test query"):
        print(f"Received: {line.strip()}")
        if "mock_event" in line:
            break


if __name__ == "__main__":
    asyncio.run(test_stream())
