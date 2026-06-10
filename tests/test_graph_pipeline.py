import asyncio

import pytest

# This suite exercises agent_utilities' knowledge_graph pipeline. The published
# agent_utilities wheel ships without that heavy subpackage, so skip the whole
# module (rather than erroring at collection) when the dev build isn't installed.
pytest.importorskip(
    "agent_utilities.knowledge_graph.pipeline.runner",
    reason="dev agent_utilities (with knowledge_graph) not installed",
)

from agent_utilities.knowledge_graph.pipeline.runner import PipelineRunner  # noqa: E402
from agent_utilities.knowledge_graph.pipeline.types import (  # noqa: E402
    PipelineContext,
    PipelinePhase,
)
from agent_utilities.models.knowledge_graph import PipelineConfig  # noqa: E402


def test_pipeline_runner_execution():
    async def _run():
        # Setup mock phases
        async def phase1_fn(ctx, deps):
            return "step1"

        async def phase2_fn(ctx, deps):
            assert deps["p1"].output == "step1"
            return "step2"

        p1 = PipelinePhase(name="p1", deps=[], execute_fn=phase1_fn)
        p2 = PipelinePhase(name="p2", deps=["p1"], execute_fn=phase2_fn)

        runner = PipelineRunner([p1, p2])
        from unittest.mock import patch

        with patch("epistemic_graph.client.SyncEpistemicGraphClient.connect"):
            ctx = PipelineContext(config=PipelineConfig(workspace_path="."))

        results = await runner.run(ctx)

        assert results["p1"].output == "step1"
        assert results["p2"].output == "step2"
        assert results["p2"].success is True

    asyncio.run(_run())


def test_pipeline_runner_failure():
    async def _run():
        async def fail_fn(ctx, deps):
            raise ValueError("Intentional failure")

        p1 = PipelinePhase(name="fail", deps=[], execute_fn=fail_fn)
        runner = PipelineRunner([p1])
        from unittest.mock import patch

        with patch("epistemic_graph.client.SyncEpistemicGraphClient.connect"):
            ctx = PipelineContext(config=PipelineConfig(workspace_path="."))

        with pytest.raises(ValueError, match="Intentional failure"):
            await runner.run(ctx)

        assert ctx.results["fail"].success is False
        assert "Intentional failure" in (ctx.results["fail"].error or "")

    asyncio.run(_run())


def test_topological_sort():
    p1 = PipelinePhase(name="p1", deps=["p2"], execute_fn=lambda _, __: asyncio.sleep(0))
    p2 = PipelinePhase(name="p2", deps=[], execute_fn=lambda _, __: asyncio.sleep(0))

    runner = PipelineRunner([p1, p2])
    # Sorted order should be p2 then p1
    assert runner.sorted_phases[0].name == "p2"
    assert runner.sorted_phases[1].name == "p1"
