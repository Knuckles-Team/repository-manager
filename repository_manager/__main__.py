import sys
from repository_manager.repository_manager_mcp import repository_manager_mcp


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        # Remove "agent" from argv so the agent's internal parser doesn't choke on it
        sys.argv.pop(1)
        from repository_manager.repository_manager_agent import agent_server

        agent_server()
    else:
        # Default to MCP or explicit 'mcp' command
        if len(sys.argv) > 1 and sys.argv[1] == "mcp":
            sys.argv.pop(1)
        repository_manager_mcp()


if __name__ == "__main__":
    main()
