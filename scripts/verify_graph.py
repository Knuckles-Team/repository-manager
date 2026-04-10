import httpx
import asyncio
import json
import sys

async def test(prompt: str = None):
    if not prompt:
        prompt = 'DEBUG: List all projects in the workspace using the repository-manager expert and the get_workspace_projects tool.'

    async with httpx.AsyncClient(timeout=120) as client:
        payload = {
            'query': prompt,
            'mode': 'ask',
            'topology': 'basic'
        }
        print(f"Connecting to http://localhost:9888/stream...")
        try:
            async with client.stream('POST', 'http://localhost:9888/stream', json=payload) as response:
                if response.status_code != 200:
                    print(f"Error: Server returned status {response.status_code}")
                    try:
                        error_data = await response.aread()
                        print(f"Detail: {error_data.decode()}")
                    except:
                        pass
                    return

                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            event_type = data.get('type')

                            # Standard graph event extraction
                            name = data.get('event') if event_type == 'graph-event' else event_type

                            if name == 'graph-start':
                                print(f"\n[GRAPH START] Run ID: {data.get('run_id')}")
                            elif name == 'node-start':
                                print(f"[NODE] {data.get('node_id')}")
                            elif name == 'tool-call' or name == 'expert_tool_call':
                                print(f"  [TOOL] {data.get('tool') or data.get('tool_name')}")
                            elif name == 'tool-result' or name == 'expert_tool_result':
                                result = str(data.get('result', ''))
                                print(f"  [RESULT] {result[:100]}...")
                            elif event_type == 'final_output':
                                print(f"\n[FINAL OUTPUT]\n{data.get('content')}")
                            elif name == 'graph-complete':
                                print(f"\n[GRAPH COMPLETE] Status: {data.get('status')}")
                            elif event_type == 'error':
                                print(f"\n[ERROR] {data.get('error')}")
                        except Exception as e:
                            pass
        except httpx.ConnectError:
            print("Error: Could not connect to the agent server. Is it running on port 9888?")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    prompt = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(test(prompt))
