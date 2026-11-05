# Seam: repository-manager governance and the meristem run loop

Provider: repository-manager RM-GOVERNANCE-R001
Consumer: meristem MERISTEM-RUN-R016

## Wire shape

No record crosses at runtime. repository-manager owns merge-queue sequencing, concept reservation, lane lifecycle and worktree operations; meristem holds no governance package and calls no repository-manager operation, and repository-manager does not call meristem. The seam is the absence of the governance modules from meristem. The consumer and the owning agents reach repository operations through the repository-manager CLI and MCP tools.

## Refusal codes

None. No call crosses the seam. A reappearance of a moved governance path in meristem fails its removal census.

## Authoritative types

repository-manager: the governance operations of `repository_manager/governance/` and `repository_manager/merge_queue.py`. meristem: the relocation inventory of its removal census.

## Test vectors

`tests/governance/test_no_parallel_lane_authority.py` holds the provider-side vector (no parallel lane authority exists); meristem holds the absence vectors in its census tests.
