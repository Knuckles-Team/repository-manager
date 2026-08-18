# Documentation readiness fleet action

`repository_manager.docs_readiness.dispatch` is the shared action core for the
RM CLI and MCP `rm_docs_readiness` tool. It consumes repository identities from
the supplied canonical `workspace.yml`; it never discovers arbitrary sibling
directories or accepts a basename/path as a repository selector.

The action has three verbs:

- `preview` (the default) calls the canonical `universal-skills` agent-readiness
  builder with `check=True` and never writes a target repository.
- `apply` requires one exact manifest-relative repository identity and an
  explicit confirmation flag. The target must be clean. The builder is called
  once to publish and again against a temporary copy to prove idempotence.
- `verify` is read-only, requires a clean target, and succeeds only when the
  canonical builder's temporary-copy output is byte-identical to the target.
  Its result carries only a provenance digest and relative generated-output
  names. The upstream `check=True` plan is not treated as an idempotence signal
  because that API reports planned paths even when their bytes are current.

The builder is loaded from the single `agent_readiness.py` resource and schema
provided by the installed `universal_skills` package. Missing or ambiguous
resources fail closed. RM does not copy or fork the generator. Applicability is
also checked before invocation (`docs/agent-readiness.json` and `mkdocs.yml`),
and generator failures are returned as bounded error codes rather than raw
exceptions.

The manifest policy explicitly excludes CI (`pipelines`, `gitlab-pipelines`),
container-image (`images/**`), deployed-service (`services/**`), and shared
skill/scaffolding identities (`agent-packages/skills/universal-skills`,
`agent-packages/skills/skill-graphs`, and `agents/tests`). Excluded identities
are reported as deliberate skips and are never probed or mutated. Every other
identity remains subject to the same exact-root, symlink, Git-cleanliness,
applicability, output-containment, and provenance checks.

Results contain logical repository identities, statuses, stable refusal codes,
relative output names, generator version, and a provenance digest. Absolute
host paths, credentials, subprocess output, and target contents never enter the
durable response.
