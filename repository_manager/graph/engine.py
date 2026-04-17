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

    async def query(self, query: str, mode: str = "hybrid") -> List[Dict[str, Any]]:
        """Multi-faceted search across in-memory NetworkX and LadybugDB."""
        if mode == "structural":
            return await self.query_structural(query)
        if mode == "semantic":
            return await self.query_semantic(query)

        # Hybrid Mode: Combine Results
        semantic_res = await self.query_semantic(query)

        # In hybrid mode, we also try a structural look for exact name matches
        # if the query is not obviously a Cypher query.
        structural_query = query
        if not query.upper().startswith(
            ("MATCH", "CREATE", "MERGE", "DELETE", "RETURN")
        ):
            structural_query = (
                f"MATCH (n) WHERE n.name = '{query}' OR n.id = '{query}' RETURN n"
            )

        structural_res = await self.query_structural(structural_query)

        # Merge and deduplicate by 'id'
        seen_ids = set()
        combined = []
        for r in semantic_res + structural_res:
            rid = r.get("id") or r.get("n", {}).get("id")
            if rid not in seen_ids:
                combined.append(r)
                if rid:
                    seen_ids.add(rid)
        return combined

    async def query_semantic(self, query_str: str) -> List[Dict[str, Any]]:
        """
        Factors in node metadata and semantic attributes from NetworkX.
        In a full implementation, this would use vector similarity.
        """
        results = []
        # Structural/Attribute fallback search in NetworkX
        for node, data in self.nx_graph.nodes(data=True):
            if (
                query_str.lower() in str(node).lower()
                or query_str.lower() in str(data.get("name", "")).lower()
            ):
                results.append({"id": node, "source": "networkx", **data})

        # If LadybugDB is connected, we could also perform a vector search here if supported
        return results

    async def query_structural(self, cypher: str) -> List[Dict[str, Any]]:
        """Executes a Cypher query on the persistent LadybugDB."""
        if not self.conn:
            return [{"error": "LadybugDB not connected."}]

        try:
            # Simple wrapper for Cypher execution
            res = self.conn.execute(cypher)
            return [dict(row) for row in res] if res else []
        except Exception as e:
            return [{"error": f"Cypher execution failed: {e}"}]

    async def query_impact(
        self, symbol: str, group_name: Optional[str] = None
    ) -> List[GraphNode]:
        """Calculates topological impact using Cypher on Ladybug with NetworkX fallback."""
        if self.conn:
            # Use Cypher to find all nodes dependent on 'symbol'
            cypher = f"MATCH (n:CodeNode {{name: '{symbol}'}})<-[*..]-(dependent) RETURN dependent"
            try:
                res = self.conn.execute(cypher)
                return [GraphNode(**dict(row)) for row in res]
            except Exception:
                pass

        # NetworkX fallback: transitive closure of predecessors
        if symbol in self.nx_graph:
            predecessors = nx.ancestors(self.nx_graph, symbol)
            return [
                GraphNode(id=p, **self.nx_graph.nodes[p])
                for p in predecessors
                if not group_name or self.nx_graph.nodes[p].get("group") == group_name
            ]
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
