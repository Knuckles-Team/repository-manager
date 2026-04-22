
import asyncio
from agent_utilities.knowledge_graph.backends import create_backend
from agent_utilities.workspace import get_agent_workspace

async def check_kg():
    ws = get_agent_workspace()
    db_path = str(ws / "knowledge_graph.db")
    print(f"Checking DB at: {db_path}")
    backend = create_backend(db_path=db_path)
    
    print("\n--- PROMPTS ---")
    res = backend.execute("MATCH (p:Prompt) RETURN p.name, p.json_blueprint")
    for row in res:
        print(f"Prompt: {row.get('p.name')} (Has Blueprint: {bool(row.get('p.json_blueprint'))})")

    print("\n--- TOOLS ---")
    res = backend.execute("MATCH (t:Tool) RETURN t.name, t.description")
    for row in res:
        print(f"Tool: {row.get('t.name')} - {row.get('t.description')}")
        
    print("\n--- RELATIONSHIPS ---")
    res = backend.execute("MATCH (s)-[r]->(t) RETURN labels(s), labels(t) LIMIT 10")
    for row in res:
        print(f"Rel: {row}")

if __name__ == "__main__":
    asyncio.run(check_kg())
