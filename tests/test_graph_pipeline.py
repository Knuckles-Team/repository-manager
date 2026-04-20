import pytest
import asyncio
import networkx as nx
from repository_manager.graph.pipeline.runner import PipelineRunner
from repository_manager.graph.pipeline.types import PipelineContext, PipelinePhase
from repository_manager.graph.pipeline.models import PipelineConfig, PhaseResult

@pytest.mark.asyncio
async def test_pipeline_runner_execution():
    # Setup mock phases
    async def phase1_fn(ctx, deps):
        return "step1"

    async def phase2_fn(ctx, deps):
        assert deps["p1"].output == "step1"
        return "step2"

    p1 = PipelinePhase(name="p1", deps=[], execute_fn=phase1_fn)
    p2 = PipelinePhase(name="p2", deps=["p1"], execute_fn=phase2_fn)

    runner = PipelineRunner([p1, p2])
    ctx = PipelineContext(
        config=PipelineConfig(workspace_path="."),
        nx_graph=nx.MultiDiGraph()
    )

    results = await runner.run(ctx)

    assert results["p1"].output == "step1"
    assert results["p2"].output == "step2"
    assert results["p2"].success is True

@pytest.mark.asyncio
async def test_pipeline_runner_failure():
    async def fail_fn(ctx, deps):
        raise ValueError("Intentional failure")

    p1 = PipelinePhase(name="fail", deps=[], execute_fn=fail_fn)
    runner = PipelineRunner([p1])
    ctx = PipelineContext(
        config=PipelineConfig(workspace_path="."),
        nx_graph=nx.MultiDiGraph()
    )

    with pytest.raises(ValueError, match="Intentional failure"):
        await runner.run(ctx)

    assert ctx.results["fail"].success is False
    assert "Intentional failure" in ctx.results["fail"].error

@pytest.mark.asyncio
async def test_topological_sort():
    p1 = PipelinePhase(name="p1", deps=["p2"], execute_fn=lambda _, __: None)
    p2 = PipelinePhase(name="p2", deps=[], execute_fn=lambda _, __: None)

    runner = PipelineRunner([p1, p2])
    # Sorted order should be p2 then p1
    assert runner.sorted_phases[0].name == "p2"
    assert runner.sorted_phases[1].name == "p1"
