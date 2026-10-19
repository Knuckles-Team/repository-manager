# CONN-REPOSITORY-MANAGER plan

## Import

The importer reuses the SDK repository transport walk (SDK-REPOSITORY-TRANSPORT-R003).
It reads commits in topological order and submits one changeset per commit through arbora.
The engine computes git object IDs (EG-SCM-R010). The importer compares each ID with the source commit.

## Mirror push

The pusher subscribes to accepted changesets of `engine_primary` repositories.
It pushes with a lease on the expected remote head. It records one receipt per remote.
The receipt shape follows RM-IDENTITY-06.

## Drift and external commits

A mirror stays writable for outside contributors. The connector imports each external commit through R001.
The engine decides acceptance. A drift check compares mirror and engine heads every minute.

## Local merge queue

`repository_manager/merge_queue.py` and `repository_manager/landing_policy.py` serve `mirror_primary` repositories.
For an `engine_primary` repository, landing moves to arbora (ARBORA-OPS-R111). Both modules refuse that repository.

## Delivery classes

- R001, R002 and R003 compose the SDK walk, the publication receipt pattern and engine methods. No mirror code exists.
- R004 wires one authority check into two existing modules.

## Migration order

The phase 2 repository migration order lives in the EG-SCM plan. This connector moves in wave 4, after the SDK.
Until then it runs the mirror for every earlier wave from its own `mirror_primary` repository.
