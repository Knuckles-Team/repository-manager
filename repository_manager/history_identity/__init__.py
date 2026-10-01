"""Fleet Git identity and history reconciliation (RM-IDENTITY-001, ``specs/fleet-git-identity``).

This package builds the **read-only, approval-gated front half** of the
fleet-wide commit-identity rewrite plan: discovery, deterministic preview, and
approval admission. It does not mutate any repository's history, refs, or
remotes — see ``specs/fleet-git-identity/spec.md`` "Decisions that must be
resolved before build or execution" for why: the public identity manifest's
unknown/external-contributor rule, signed-object re-signing policy, and
per-remote protected-branch permissions are owner decisions the spec records
as still open. Building the actual quarantined rewrite, backup-ref
transaction, lease-protected remote publication, and merge-queue
coordination (RM-IDENTITY-04 through -07) on top of an unresolved policy
would bake in a guess where the spec requires an explicit, reviewed answer.

What *is* buildable without those decisions, because it only reasons about an
injected policy rather than deciding what the policy should contain:

* :mod:`.discovery` (RM-IDENTITY-01) — enumerate every ref, tag, and
  remote-tracking ref in a repository and refuse when discovery is
  incomplete or the repository state is divergent.
* :mod:`.policy` (part of RM-IDENTITY-02) — an ``IdentityPolicy`` model with a
  deterministic digest, built from aliases the caller supplies and a
  canonical-identity allowlist loaded from the single external file the
  fleet's commit-identity gate already owns (``pipelines_hooks.identity``) —
  this package carries no second copy of that list.
* :mod:`.preview` (RM-IDENTITY-02) — a deterministic dry-run plan over a
  discovery result and a policy: old->new identity mapping, counts, and a
  plan digest that is stable across repeated runs and changes whenever the
  source or policy changes.
* :mod:`.approval` (RM-IDENTITY-03) — fail-closed verification that a
  purported approval is bound to the exact repository set, source digest,
  policy digest, remotes, and an unexpired expiry.
* :mod:`.consistency` (RM-IDENTITY-R002) — the general "is this repository's
  branch history coherent enough to admit to a rewrite plan" check; applying
  it to a specific repository (the spec names ``agent-webui``) is an
  operational step outside this package.

RM-IDENTITY-R001 (the end-to-end rewrite) and RM-IDENTITY-04 through -07
(byte-identical tree rewrite, quarantine/backup refs, lease-protected
publication, merge-queue coordination) are not attempted here.
"""

from __future__ import annotations
