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
  once to publish and then verified through a bounded staging output directory
  using the target's exact source inputs; RM never copies the repository tree.
  RM snapshots only the bounded generator-owned output namespace before apply
  and restores it if publication or the post-apply verification fails.
- `verify` is read-only, requires a clean target, and succeeds only when the
  canonical builder's bounded staging output is byte-identical to the target's
  governed artifacts. RM snapshots Git status and those artifact paths before
  and after the call so a non-conforming generator cannot mutate a source file
  or target artifact silently.
  Its result carries only a provenance digest and relative generated-output
  names. The upstream `check=True` plan is not treated as an idempotence signal
  because that API reports planned paths even when their bytes are current.

The builder is loaded from the single `agent_readiness.py` resource and schema
provided by the installed `universal_skills` package. Missing or ambiguous
resources fail closed. RM does not copy or fork the generator. Applicability is
also checked before invocation (`docs/agent-readiness.json` and `mkdocs.yml`),
and generator failures are returned as bounded error codes rather than raw
exceptions. Each repository must have its readiness configuration generated or
explicitly adopted before rollout; this action does not create that
configuration or make every repository rollout-ready.

The default fleet is the exact set of manifest-declared identities beneath
`agent-packages/` (75 publishable entries in the current manifest), including
both shared skill repositories. Only `agent-packages/agents/tests` is
non-publishable when that identity is present. RM fails closed if the pinned
publishable count changes or a selected checkout is missing; it does not
silently process a partial fleet. No filesystem discovery expands or changes
the selection. Every selected identity remains subject to the same exact-root,
symlink, Git-cleanliness, applicability, output-containment, and provenance
checks. Unlisted sibling directories are ignored rather than promoted into the
fleet.

Results contain logical repository identities, statuses, stable refusal codes,
relative output names, generator version, and a provenance digest. Absolute
host paths, credentials, subprocess output, and target contents never enter the
durable response.
