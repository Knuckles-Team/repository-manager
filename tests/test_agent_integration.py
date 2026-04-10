import os
import sys
import time
import subprocess
import httpx
import json
import pytest
import signal
from typing import Generator

# Configuration
PORT = 9888
BASE_URL = f"http://localhost:{PORT}"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", ".."))

@pytest.fixture(scope="module")
def agent_server() -> Generator[subprocess.Popen, None, None]:
    """Starts the agent server and shuts it down after tests."""
    env = os.environ.copy()
    # Ensure REPOSITORY_MANAGER_WORKSPACE points to the real workspace
    env["REPOSITORY_MANAGER_WORKSPACE"] = WORKSPACE_ROOT
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "repository_manager", "agent_server.py"),
        "--port", str(PORT),
        "--debug"
    ]

    print(f"\nStarting Agent Server: {' '.join(cmd)}")
    # Ensure port is clear
    try:
        subprocess.run(["lsof -i :9888 -t | xargs kill -9"], shell=True, capture_output=True)
    except:
        pass
    time.sleep(2) # Wait for port to clear

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )

    # Wait for server to be healthy
    max_retries = 30
    for i in range(max_retries):
        try:
            resp = httpx.get(f"{BASE_URL}/health", timeout=1.0)
            if resp.status_code == 200:
                print(f"Server is healthy after {i} seconds.")
                break
        except Exception:
            pass

        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"Server failed to start:\nSTDOUT: {stdout.decode()}\nSTDERR: {stderr.decode()}")

        time.sleep(1)
    else:
        process.terminate()
        pytest.fail("Server health check timed out.")

    yield process

    # Shutdown
    print("\nShutting down Agent Server...")
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    process.wait(timeout=10)

@pytest.mark.asyncio
async def test_get_workspace_projects_via_graph(agent_server):
    """Verifies that the agent server correctly orchestrates a call to get_workspace_projects."""
    query = "List all projects in the workspace."

    async with httpx.AsyncClient(timeout=60.0) as client:
        # We use the /stream endpoint which is designed for high-fidelity graph execution
        # This endpoint returns an SSE stream with granular events
        url = f"{BASE_URL}/stream"
        payload = {
            "query": query,
            "mode": "ask",
            "topology": "basic"
        }

        print(f"Sending query to {url}: {query}")

        graph_started = False
        tool_called = False
        final_output_received = False

        try:
            async with client.stream("POST", url, json=payload) as response:
                assert response.status_code == 200

                async for line in response.aiter_lines():
                    if not line or not line.strip():
                        continue

                    if line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                            print(f"DEBUG: Received event: {data}")  # NEW
                            etype = data.get("type")
                            ename = data.get("event")

                            if etype == "graph-event":
                                if ename == "graph-start":
                                    graph_started = True
                                elif ename == "expert_tool_call" or ename == "tool-call":
                                    if data.get("tool_name") == "get_workspace_projects" or data.get("tool") == "get_workspace_projects":
                                        tool_called = True
                                elif ename == "graph-complete":
                                    final_output_received = True

                            if etype == "final_output":
                                content = data.get("content", "")
                                if "agent-packages" in content or "repository-manager" in content:
                                    final_output_received = True
                        except Exception as e:
                            print(f"DEBUG: Error parsing line: {line} - {e}") # NEW
                            pass
        except Exception as e:
            pytest.fail(f"SSE request failed: {e}")

    assert graph_started, "The graph orchestrator did not start"
    assert tool_called, "The graph orchestrator did not call get_workspace_projects"
    assert final_output_received, "The final response did not contain the expected project list"

    # Fallback: Check direct MCP execution via another endpoint if ag-ui is purely chat
    # Actually, we want to see if the graph reached the tool.

    assert tool_called, "The graph orchestrator did not call get_workspace_projects"
    assert final_output_received, "The final response did not contain the expected project list"

if __name__ == "__main__":
    # Manual execution helper
    import json
    import asyncio

    async def run_manual():
        try:
            # We can't use the fixture easily here, so just test a running server
            print("Running manual verification against localhost:9888...")
            query = "What projects are in the workspace?"
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", f"{BASE_URL}/ag-ui", json={"message": query}) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            print(line[6:])
        except Exception as e:
            print(f"Error: {e}")

    # If port is open, run manual
    try:
        httpx.get(f"{BASE_URL}/health")
        asyncio.run(run_manual())
    except:
        print("Server not running. Run with pytest.")
