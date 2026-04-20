import pickle
import networkx as nx

with open("/home/genius/Workspace/.repo_graph/graph.pkl", "rb") as f:
    graph = pickle.load(f)

print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")

# Look for CALLS edges
calls = [(u, v) for u, v, d in graph.edges(data=True) if d.get("type") == "CALLS"]
print(f"CALLS edges: {len(calls)}")
if calls:
    print(f"Example: {calls[0]}")
    target = calls[0][1]
    ancestors = nx.ancestors(graph, target)
    print(f"Ancestors of {target}: {ancestors}")
