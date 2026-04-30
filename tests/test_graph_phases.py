import networkx as nx
import pytest
from agent_utilities.knowledge_graph.pipeline.phases.parse import execute_parse
from agent_utilities.knowledge_graph.pipeline.phases.resolve import execute_resolve
from agent_utilities.knowledge_graph.pipeline.phases.scan import execute_scan
from agent_utilities.knowledge_graph.pipeline.types import PipelineContext
from agent_utilities.models.knowledge_graph import PhaseResult, PipelineConfig


@pytest.mark.asyncio
async def test_scan_and_parse_integration(tmp_path):
    # Create dummy files
    py_file = tmp_path / "test.py"
    py_file.write_text("class MyClass:\n    pass\n\ndef my_func():\n    pass")

    ctx = PipelineContext(
        config=PipelineConfig(workspace_path=str(tmp_path)), nx_graph=nx.MultiDiGraph()
    )

    # Run scan
    files = await execute_scan(ctx, {})
    assert len(files) >= 1
    assert any("test.py" in f for f in files)

    # Run parse
    scan_result = PhaseResult(name="scan", duration_ms=0, output=files)
    parse_output = await execute_parse(ctx, {"scan": scan_result})

    assert parse_output["symbols_extracted"] >= 2
    # Verify nodes in graph
    nodes = list(ctx.nx_graph.nodes())
    assert any("symbol:MyClass" in n for n in nodes)
    assert any("symbol:my_func" in n for n in nodes)


@pytest.mark.asyncio
async def test_resolve_imports(tmp_path):
    # Create two files, one importing from another
    file_a = tmp_path / "a.py"
    file_a.write_text("from b import MyClass")
    file_b = tmp_path / "b.py"
    file_b.write_text("class MyClass: pass")

    ctx = PipelineContext(
        config=PipelineConfig(workspace_path=str(tmp_path)), nx_graph=nx.MultiDiGraph()
    )

    # Scan & Parse
    files = await execute_scan(ctx, {})
    scan_res = PhaseResult(name="scan", duration_ms=0, output=files)
    await execute_parse(ctx, {"scan": scan_res})

    # Resolve
    parse_res = PhaseResult(name="parse", duration_ms=0, output={})
    resolve_output = await execute_resolve(ctx, {"parse": parse_res})

    assert resolve_output["resolved_dependencies"] >= 1
    # Verify edge exists between file a and file b
    edges = list(ctx.nx_graph.edges(data=True))
    assert any(d.get("type") == "depends_on" for u, v, d in edges)
