# Seam: migration wave entry criteria

Provider: pipelines PIPE-GATES-R006
Consumer: repository-manager RM-SCM-R004.2

## Wire shape

A wave starts only when its entry criteria hold and the prior wave has exited; the wave step reads the criteria of the wave it is asked to start and refuses when one is unmet.

## Refusal codes

`WAVE_ENTRY_UNMET`, raised by the connector.

## Authoritative types

The wave definitions of `PIPE-GATES-R006` and its children.

## Test vectors

`tests/source_changesets/test_wave_step.py` (planned) holds the connector-side vector.
