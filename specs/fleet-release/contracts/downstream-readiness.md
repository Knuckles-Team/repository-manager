# Seam: downstream readiness wheel rule

Provider: pipelines PIPE-RELEASE-R008
Consumer: repository-manager RM-RELEASE-R005

## Wire shape

The consumer reads the package index and judges each dependency version by the pipelines wheel rule: a version is available when it has a wheel compatible with the target interpreter or platform. A missing sdist is the advisory `sdist_missing`. A version with only an sdist, or with no compatible wheel, fails unless the consumer lists the package in `[tool.pipelines] allow-source-build`. `partial_publish` is reported only while the forge reports the version's release run as in progress.

## Refusal codes

`sdist_missing` (advisory), `partial_publish` (transient failure) and the failure for a version with no compatible wheel.

## Authoritative types

`AvailableVersions` and `ConstraintCheck` in `repository_manager/dependency_readiness.py`.

## Test vectors

`tests/test_dependency_readiness.py` holds the consumer-side vectors.
