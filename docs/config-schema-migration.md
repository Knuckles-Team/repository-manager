# Repository declaration schemas and migration

Repository Manager accepts two repository-owned declarations:

- `.buildcache.yaml` describes build argv, cache inputs, artifacts, and the
  resource/placement contract that a later admission scheduler will consume.
- `.mergequeue.yaml` describes gate argv, validation stage, differential baseline,
  path selection, resource requirements, artifact dependencies, and generated-file
  regeneration.

Both declarations use `schema_version: 2`. The historical unversioned shape is
read as version 1 during the compatibility window and is normalized in memory;
it is never treated as an implicit safety default. Version-2 unknown keys fail
closed. Version-1 unknown keys produce `ConfigCompatibilityWarning` naming the
file and field, so a migration cannot silently lose a safety setting.

## Build declaration

```yaml
schema_version: 2
base: main
specs:
  - name: frontend
    command: [pnpm, build]
    workdir: .
    toolchain_fingerprint: [bash, -c, "node --version && pnpm --version"]
    cache_key_paths: [src, package.json, pnpm-lock.yaml]
    artifact_contract:
      patterns: [dist/**/*]
      required: true
      publish: true
      retention: content-addressed
    resource_class: frontend-build
    resources:
      cpu_weight: 8
      memory_mb: 8192
      disk_mb: 4096
      process_slots: 1
    placement:
      required_labels: [nodejs, pnpm]
      anti_affinity: [frontend-build]
    stage: certification
    generation_compatible: true
    timeout: 900
```

`artifacts` remains accepted as a compact compatibility spelling for
`artifact_contract.patterns`. `disk_estimate_mb` is also accepted and must
agree with `resources.disk_mb` when both are present. Workdirs, cache paths,
artifact patterns, and gate path globs are relative and cannot contain `..`.
Commands are argv lists; shell strings, empty argv, negative resource values,
and malformed glob/regex patterns are refused before any job is admitted.

## Merge declaration

```yaml
schema_version: 2
base: main
batch_size: 8
environment_signature: [python3, --version]
gates:
  - name: tests
    command: [python3, -m, pytest, tests]
    stage: integration
    baseline_mode: differential
    path_selection:
      include: ["**/*.py"]
      exclude: ["**/generated/**"]
    resource_class: light-check
    resources: {cpu_weight: 2, memory_mb: 2048, process_slots: 1}
    artifact_dependencies: [wheel]
    timeout: 300
    baseline_timeout: 600
    compare: pytest-ids
    on_timeout: defer

generated_files: [README.md]
regenerate: [[python3, scripts, regenerate_docs.py]]
```

The v1 `tier: fast` and `tier: slow` spellings map deterministically to
`stage: integration` and `stage: certification`, respectively, with a
deprecation warning. The queue still receives its existing `tier` projection
from the typed parser, so this lane does not change gate execution policy.
`stage: smoke` and `stage: release` are metadata for later validation/release
lanes and project to the conservative slow tier until those lanes take over.

The known drifted shape is migrated as well:

```yaml
regenerate_on_conflict:
  paths: [README.md]
  regenerate: "python3 -c \"print('generator')\""
```

The command is parsed with `shlex.split` into argv and becomes
`generated_files`/`regenerate`. No shell is executed during parsing or
migration. If both old and new forms are present, their files and commands are
unioned deterministically rather than silently dropping one declaration.

## Preview, apply, and rollback

The migration API is read-only by default:

```python
from repository_manager.config_migration import (
    apply_migration,
    preview_migration,
    rollback_migration,
)

preview = preview_migration(".mergequeue.yaml")  # diff only; no write
applied = apply_migration(".mergequeue.yaml")    # atomic replace + .bak backup
rollback_migration(".mergequeue.yaml", applied)  # guarded restoration
```

Applying an already-versioned canonical file is a byte-stable no-op. A changed
file is written through a same-directory temporary file and `os.replace` after
the original is copied to `<config>.bak`. Rollback checks that the current file
still has the applied target digest; it refuses to overwrite operator edits.
The parser and preset validator use the same schema functions as migration, so a
preview cannot validate a shape that runtime loading would reject.

Presets remain templates, not an installation step. Validate the packaged
presets with `validate_presets()` before a later cutover lane installs them in a
target repository.
