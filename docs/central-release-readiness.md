# Central RELEASE hook rollout

Pipelines owns the shared hook catalogue, manual stage, exact-wheel profile
selection and publication proof. This repository owns fleet readiness and the
existing updater. Consumers link the [central release contract](https://github.com/Knuckles-Team/pipelines/blob/main/docs/python-release-readiness.md)
instead of copying policy. Immutable workflow/hook pins remain caller wiring.

The new `repository_manager.release_readiness_hook` delegates to the existing
checker while rejecting missing/invalid fleet scope and overridden results.
It does not replace wheel proof, runtime admission or graph attachment checks.
A prepared interpreter must include this entry point; missing installations
block, including outside CI. No checker is downloaded by the shared hook.

Run the existing sweep with a locally available, reviewed pipelines commit:

```sh
python scripts/sweep_dependency_readiness_hook.py --root /fleet \
  --pipelines-checkout /checkouts/pipelines --pipelines-ref FULL_COMMIT_SHA --dry-run
```

The diff removes only recognized legacy shell entries and adds a pinned shared
hook ID. It preserves other hooks, top-level options and formatting. Custom
entries, aliases, duplicates, comments within replaced entries and malformed
configs require manual review. Existing central entries at the requested pin
are idempotent; customized or differently pinned entries require review.

Before applying, audit every actual publisher (including alternate actions and
inline scripts): each package publisher must be pinned to the guarded workflow,
and downstream runtime image publishers must depend on its success. The sweep
cannot establish arbitrary workflow semantics; `--apply --publishers-reviewed`
records the operator's explicit prerequisite acknowledgment. It does not edit
publisher refs, consumer guidance, runtime environments or profile inputs.
Prepare RM via a pinned hook `additional_dependencies` reference or the shared
hook's `--python` input. Any missing runtime remains a blocking result.

Use ordinary source tests with the pinned siblings without removing dependency
floors. Release/index gates may remain blocked by unpublished dependencies.
No hook stage change, provider declaration or passing source test constitutes
publication or runtime attachment evidence.
