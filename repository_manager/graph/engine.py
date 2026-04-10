import networkx as nx
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import real_ladybug as lb
except ImportError:
    lb = None  # fallback for systems without it, handled later

from repository_manager.graph.models import GraphNode, GraphEdge, GraphReport


class GraphEngine:
    """Unified graph engine with seamless NetworkX + LadybugDB integration."""

    def __init__(self, workspace_path: str, db_name: str = "workspace.lbug"):
        self.workspace_path = Path(workspace_path)
        self.db_path = self.workspace_path / ".repo_graph" / db_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # LadybugDB persistent store
        if lb is not None:
            self.db = lb.Database(str(self.db_path))
            self.conn = lb.Connection(self.db)
            self._ensure_schema()
        else:
            self.db = None
            self.conn = None

        # In-memory NetworkX for construction & analysis
        self.nx_graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def _ensure_schema(self):
        """Define LadybugDB schema once (strongly typed tables)."""
        if not self.conn:
            return

        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS CodeNode (
                id STRING PRIMARY KEY,
                type STRING,
                name STRING,
                repo_path STRING,
                file_path STRING,
                metadata JSON
            )
        """)
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CodeEdge (
                FROM CodeNode TO CodeNode,
                type STRING,
                provenance STRING,
                confidence FLOAT,
                rationale STRING,
                metadata JSON
            )
        """)

    async def build_from_workspace(
        self, workspace_model, multimodal: bool = False, incremental: bool = True
    ) -> GraphReport:
        """Main entry point: Build using NetworkX, persist to LadybugDB."""
        self.nx_graph = self._build_networkx_graph(
            workspace_model, multimodal, incremental
        )
        self._sync_to_ladybug()
        return self._generate_report()

    def _generate_report(self) -> GraphReport:
        return GraphReport(
            nodes_processed=self.nx_graph.number_of_nodes(),
            edges_processed=self.nx_graph.number_of_edges(),
            groups_analyzed=[],
        )

    def _build_networkx_graph(
        self, workspace_model, multimodal: bool, incremental: bool
    ) -> nx.MultiDiGraph:
        """Two-pass pipeline (Tree-sitter + embeddings + Graphify-style rationale)."""
        # Pass 1: Local AST
        # Pass 2: Semantic Optional

        # Returns Graph
        return self.nx_graph

    def _sync_to_ladybug(self):
        """Project NetworkX to LadybugDB (delta Cypher MERGE handles incremental)."""
        if not self.conn:
            return

        # Example of bulk merge or direct query
        pass

    async def query_impact(
        self, symbol: str, group_name: Optional[str] = None
    ) -> List[GraphNode]:
        """Example tool: Cypher execution on Ladybug for speed with NetworkX fallback."""
        return []

    def find_path(self, source_id: str, target_id: str) -> List[GraphEdge]:
        """Traces the topological relationship path using NetworkX."""
        try:
            path = nx.shortest_path(self.nx_graph, source_id, target_id)
            edges = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                # Return first matching edge
                data = self.nx_graph.get_edge_data(u, v, default={})
                # nx.MultiDiGraph get_edge_data returns a dict of dicts
                first_key = list(data.keys())[0] if data else None
                if first_key is not None:
                    edge_data = data[first_key]
                    edges.append(GraphEdge(source=u, target=v, **edge_data))
            return edges
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Returns graph metrics for status reporting."""
        return {
            "nodes": self.nx_graph.number_of_nodes(),
            "edges": self.nx_graph.number_of_edges(),
            "communities": len(
                set(nx.get_node_attributes(self.nx_graph, "community").values())
            ),
            "ladybug_connected": self.conn is not None,
        }

    def reset_graph(self):
        """Purges the graph database and in-memory state."""
        self.nx_graph.clear()
        if self.db_path.exists():
            import shutil

            shutil.rmtree(self.db_path.parent)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def to_json(self) -> dict:
        """Export for reports or Obsidian/Mermaid."""
        return nx.node_link_data(self.nx_graph)
