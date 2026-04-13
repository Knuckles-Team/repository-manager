import httpx
import asyncio
import json
import sys
import os
import subprocess
import time
from typing import List, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DEFAULT_PORT = 9888
SERVER_URL = f"http://localhost:{DEFAULT_PORT}"

def get_ground_truth() -> List[str]:
    """Get the ground truth projects list directly from the logic."""
    print("Fetching ground truth from Git logic...")
    try:
        from repository_manager.mcp_server import get_git_instance
        git = get_git_instance()
        projects = list(git.project_map.keys())
        print(f"Ground Truth: {len(projects)} projects found.")
        return projects
    except Exception as e:
        print(f"Error fetching ground truth: {e}")
        return []

async def wait_for_server(url: str, timeout: int = 30):
    """Wait for the agent server to be responsive."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{url}/health")
                if resp.status_code == 200:
                    print("Server is up and healthy!")
                    return True
        except Exception:
            pass
        await asyncio.sleep(1)
    return False

async def test(prompt: str = None):
    if not prompt:
        # SIMPLER PROMPT
        prompt = 'List all projects in the workspace.'

    ground_truth = get_ground_truth()
    
    server_process = None
    # Check if anything is on port
    try:
        async with httpx.AsyncClient() as client:
            await client.get(SERVER_URL, timeout=1)
            print(f"Port {DEFAULT_PORT} already occupied. Assuming server is running.")
    except Exception:
        print(f"Starting agent server on port {DEFAULT_PORT}...")
        env = os.environ.copy()
        # DON'T HIDE OUTPUT
        server_process = subprocess.Popen(
            [sys.executable, "-m", "repository_manager.agent_server", "--port", str(DEFAULT_PORT)],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        if not await wait_for_server(SERVER_URL):
            print("Timeout waiting for server to start.")
            if server_process:
                server_process.terminate()
            return

    captured_projects = []
    
    async with httpx.AsyncClient(timeout=600) as client:
        payload = {
            'query': prompt,
            'mode': 'ask',
            'topology': 'basic'
        }
        print(f"Connecting to {SERVER_URL}/stream...")
        try:
            async with client.stream('POST', f'{SERVER_URL}/stream', json=payload) as response:
                if response.status_code != 200:
                    print(f"Error: Server returned status {response.status_code}")
                    return

                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        try:
                            raw_data = json.loads(line[6:])
                            
                            # DEBUG: Print ALL events
                            # print(f"  [DEBUG EVENT] {raw_data.get('type') or raw_data.get('event')}")
                            
                            data = raw_data
                            name = None
                            
                            if data.get('type') == 'data-graph-event':
                                data = data.get('data', {})
                                name = data.get('event')
                            else:
                                name = data.get('event') if data.get('type') == 'graph-event' else data.get('type')

                            if name == 'graph-start':
                                print(f"\n[GRAPH START] Run ID: {data.get('run_id')}")
                            elif name == 'node-start':
                                print(f"\n[NODE] {data.get('node_id')}")
                            elif name == 'agent-node-delta':
                                content = data.get('content', '')
                                print(content, end='', flush=True)
                            elif name in ['tool-call', 'expert_tool_call']:
                                print(f"  [TOOL] {data.get('tool') or data.get('tool_name')}")
                            elif name in ['tool-result', 'expert_tool_result']:
                                result = data.get('result')
                                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], str) and result[0].startswith('http'):
                                    captured_projects = result
                                print(f"  [RESULT] {str(result)[:100]}...")
                            elif name == 'routing_completed':
                                plan = data.get('plan', {})
                                steps = plan.get('steps', [])
                                print(f"  [ROUTER] Generated plan with {len(steps)} steps: {[s.get('node_id') for s in steps]}")
                            elif name == 'final_output' or raw_data.get('type') == 'final_output':
                                content = data.get('content') or raw_data.get('content')
                                print(f"\n[FINAL OUTPUT]\n{content}")
                            elif name == 'graph-complete':
                                print(f"\n[GRAPH COMPLETE] Status: {data.get('status')}")
                            elif raw_data.get('type') == 'error':
                                print(f"\n[ERROR] {raw_data.get('error')}")
                        except Exception as e:
                            pass
        except Exception as e:
            import traceback
            print(f"An error occurred during streaming: {e}")
            traceback.print_exc()
        finally:
            if server_process:
                print("Terminating agent server...")
                server_process.terminate()

    # Parity Check
    print("\n" + "="*40)
    print("PARITY CHECK")
    print("="*40)
    
    if not captured_projects:
        print("❌ FAILED: No projects captured from tool execution.")
    else:
        # Compare sets
        ground_set = set(ground_truth)
        captured_set = set(captured_projects)
        
        matches = ground_set == captured_set
        
        print(f"Ground Truth Count: {len(ground_set)}")
        print(f"Captured Count:     {len(captured_set)}")
        
        if matches:
            print("\n✅ Integration Validated: Tool output exactly matches ground truth.")
        else:
            diff_ground = ground_set - captured_set
            diff_captured = captured_set - ground_set
            print("\n❌ Integration Mismatch Found!")
            if diff_ground:
                print(f"Missing in Captured: {list(diff_ground)[:5]}...")
            if diff_captured:
                print(f"Extra in Captured:   {list(diff_captured)[:5]}...")
    print("="*40)

if __name__ == '__main__':
    prompt = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(test(prompt))
