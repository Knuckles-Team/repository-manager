import sys
from repository_manager.mcp_server import repository_manager_mcp


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        sys.argv.pop(1)
        from repository_manager.agent_server import agent_server

        agent_server()
    else:
        if len(sys.argv) > 1 and sys.argv[1] == "mcp":
            sys.argv.pop(1)
        repository_manager_mcp()


if __name__ == "__main__":
    main()
