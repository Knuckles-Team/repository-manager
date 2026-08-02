# The conflict-resolution decision procedure

> The long form of the decision tree in `SKILL.md`. Load this when you are
> actually holding a conflict, not to orient.

The governing question is never "how do I make the markers go away." It is:

> **What tree do I want to exist after this lands, and what does each side
> believe it is protecting?**

Run the branches top-down and stop at the first match.

---

## Step 0 — establish what you are even looking at

Before anything else, materialize the tree that will actually exist. The branch
tip is not it, and reasoning from the tip has already misled three people here in
a single day — one of whom concluded a branch had deleted a guard that the merged
tree kept, and raised a false session-wide alarm from `git show <branch>:<path>`.

```bash
TREE=$(git merge-tree --write-tree <base> <branch>)   # exit 0 = clean, 1 = conflicts
git ls-tree -r "$TREE" -- <path>                      # what the file will look like
git diff <base> "$TREE" -- <path>                     # what the merge actually changes
```

Two facts to write down before proceeding:

1. **Which paths conflict**, and for each: is it authored or derived?
2. **How far the base has moved** — `git rev-list --count $(git merge-base HEAD <base>)..<base>`.

`repository-manager --lane doctor --lane-path .` reports the second as
`base-drift` automatically.

---

## Branch 1 — the conflict is only in DERIVED files → regenerate

**Signature:** every conflicting path is produced by a generator: a lockfile, a
folded ledger view (`docs/concept_reservations.yaml`, `reports/PROGRAM.md`), a
generated manifest, an API/docs index, a coverage or inventory table.

At merge-queue scale this is the *majority* case. With ~76 candidates queued on
one base, nearly every one conflicts on a purely-derived file where there is no
real disagreement at all — both sides regenerated the same artifact from
different inputs.

**Do:**

```bash
# 1. take the merged tree as it stands (inputs from BOTH sides)
# 2. re-run the generator IN that tree
python3 scripts/<generator>.py
git add <generated path>
```

**Do not:**

- `git checkout --theirs` / `--ours` on a generated file. It does not drop one
  side's *output*; it drops one side's **input**. The regenerated artifact will
  silently omit a lane's contribution and nothing will fail. This is a data-loss
  bug wearing a conflict-resolution costume.
- Hand-edit the generated view to splice both halves together. `lane-guard`
  refuses a hand-edited ledger view for exactly this reason, and a hand-merged
  fold has already needed repair by hand-verified line-union.

The merge queue does this automatically on land, driven by
`generated_files:` and `regenerate:` in the repository's `.mergequeue.yaml`. If
you are resolving by hand, you are reproducing that step — declare the file in
`.mergequeue.yaml` afterwards so nobody has to do it by hand again.

**Sub-case — append-only fragment stores.** If the "generated" file is a *fold*
of per-lane fragments (`docs/concept_reservations.d/<lane>.yaml`,
`reports/deferred/<lane>.md`), there is no conflict to resolve at all: your
fragment and theirs are different files. Take the union of the fragment
directory and re-fold. If you find yourself resolving a conflict *inside* the
folded view, you are editing the wrong file.

---

## Branch 2 — the base moved since you measured → re-measure

**Signature:** your evidence (a test run, a diff, a file read) predates commits
that have since landed on the base.

Everything you concluded describes a tree that will never exist. Do not resolve
anything yet.

```bash
git fetch origin
git merge-tree --write-tree <base> HEAD      # recompute
```

Re-run whatever measurement produced your conclusion **against that tree**, then
re-enter this procedure from Step 0. Very often the conflict you were about to
resolve is no longer there, or is a different conflict entirely.

Do **not** rebase to "fix" this reflexively. Rebasing rewrites your branch's
history under other lanes that may have already read it, and it re-runs every
conflict once per commit instead of once. Merge the base *into your branch* and
resolve there:

```bash
git merge origin/<base>       # resolve ON YOUR BRANCH, never in the canonical tree
```

---

## Branch 3 — a gate is red → is the failure NEW or PRE-EXISTING?

**Signature:** no text conflict; the merge is clean, but a gate refuses it.

`main` here is legitimately red. Treating red-as-blocking absolutely has already
deadlocked the queue and **stranded 19 branches**; a branch that fixed 21 of 30
failing tests was rejected because 9 remained. So the only meaningful question
is whether *this candidate* introduced the signal.

```bash
# run the SAME gate command, twice, in two throwaway trees
git worktree add --detach /tmp/base-probe <base>
git worktree add --detach /tmp/merged-probe "$TREE"
# ... run the gate in each ...
```

Then compare at the granularity the gate declares:

| `compare` | How to compare | Trap |
|---|---|---|
| `pytest-ids` | set of failing **node ids**. A failing id passes only if that exact id fails on the base. | Never compare by file, module, pattern, or *count*. Count-based comparison is trivially gamed: delete one failing test, add a worse one, count unchanged. An exit code outside `{0,1,5}` is **unreadable**, not "zero failures" — refuse it. |
| `lines` | set of normalized output lines, filtered by `keep_lines` / `ignore_lines`. Substitute the tree path out first — the two runs happen in two different throwaway worktrees, so every absolute path differs. | Without `keep_lines`, a chatty tool's per-run noise (`Compiling foo v0.1.0`, `Finished in 3.4s`) reads as a new violation on **every** candidate. |
| `exit` | the exit code only. | Report exactly that much precision. A tool that prints one static message cannot tell you *why*; do not narrate a cause you did not observe. |

**Outcomes:**

- **Present on the base too** → pre-existing. Not yours. Record it in the
  register; do not expand your branch's scope to fix it. (Do not *hide* it
  either — a pre-existing failure you noticed is worth a register entry.)
- **Absent on the base** → new. Yours. Fix on your branch and re-enqueue.
- **The baseline could not be produced** (the base ref does not build, the gate
  times out, the environment cannot be fingerprinted) → ★ **REFUSE the
  candidate.** An unknown must not be spelled the same way as a pass. Fix the
  baseline first. "Allow everything through because we could not measure" is how
  a gate becomes decorative.

★ Before you trust a *green* gate at all: **prove it catches a deliberately
introduced known-bad input.** Three gates here were found green while enforcing
nothing — one crashing, one blind to 2 of 16 patterns it claimed to cover, one
never discovered by its own runner. Break the thing on purpose, watch the gate
refuse it, revert. A gate that has only ever seen good input has not been tested.

---

## Branch 4 — ★ SEMANTIC DIVERGENCE inside a textual conflict

**Signature:** a real conflict in authored code where both sides compile, both
sides' tests pass, and the hunks look interchangeable.

This is the dangerous one, because every automated and semi-automated resolution
strategy gets it wrong, and nothing downstream will catch it.

### The incident this rule comes from

Two lanes touched the same Cypher query builder. Lane A had replaced a literal
string splice with a **bound parameter** — an injection fix. Lane B had, in the
same region, changed the query for an unrelated feature and still had the literal
splice. The conflict markers showed two plausible-looking query constructions.

Taking lane B's side would have compiled, passed every test, landed, and
**silently reverted a security fix**. No gate would have noticed: the tests
covered the query's *results*, not how it was built.

### The procedure

1. **For each side, answer "what invariant is this side holding?"** — not "what
   does this line say." Read the commit message and, decisively, the **test that
   arrived with it**. A test is the most reliable statement of intent in the
   repository.

   ```bash
   git log --oneline <base>..<branch> -- <path>
   git log -1 --format=%B <sha>
   git show <sha> --stat            # what test came with it?
   ```

2. **Classify each side.** If either side is a **security**, **correctness**, or
   **resource-safety** fix, that invariant is not negotiable. Keep it, and
   re-apply the other side's feature *on top of* it. A feature can always be
   re-expressed; a silently reverted fix cannot be noticed.

3. **Prove the merged result satisfies BOTH sides' tests.** Not "the suite is
   green" — specifically both sides' tests, run against the merged tree.

4. **If one side shipped a defect-pinning test, verify it is not vacuous.**
   Restore the bug it claims to pin and confirm the test **fails**. A lane here
   caught its own test passing against the very bug it claimed to pin; another
   found a gate meta-test that had encoded a bug *as correct*. A pinning test
   that has never been observed failing is not evidence.

5. **If both sides look equally reasonable, stop and get a second reader.** That
   symmetry is the signal, not an invitation to choose. The cheapest moment to
   involve someone else is before the resolution lands, not after it silently
   removes something.

### Finding every consumer before you resolve

If the conflict is in a contract (a signature, a schema, a tool name), you must
know every caller — and greps miss them:

- `import x as y` — aliased, the name never appears
- `from … import x` — bare, no module prefix
- `monkeypatch.setattr("pkg.mod.x", …)` — a *string*, invisible to symbol search

All three have bitten this workspace. Use the AST-based
`scripts/find_callers.py`, not grep.

And when you wire a control, wire it at the **chokepoint**, not one entrypoint.
A control wired at a single entrypoint was deployed here and changed literally
nothing, because six callers bypassed it.

---

## Branch 5 — same intent, genuinely textual → resolve, then prove

Both sides mean the same thing and only the text collides (two lines added to the
same list, adjacent edits to one function). Resolve to hold both intentions.

Then **prove it**: run the affected gate against the merged tree. A resolution
that was never executed is a hypothesis, and merge conflicts are exactly where
plausible-looking code goes unexecuted.

---

## After the resolution

```bash
git add -A && git commit                 # never --no-verify
repository-manager --lane doctor --lane-path .
repository-manager --merge-queue enqueue --repo-path .
```

The scheduler lands it within ~5 minutes and prunes the worktree and the branch.

**Never `git branch -D`.** `-d`'s refusal is the safety mechanism telling you the
work is not actually contained in the base — usually because the resolution
landed as a *different* commit than the branch tip. Investigate the refusal
rather than overriding it.

**Never hand-merge into the shared base to skip the queue.** Two lanes
hand-merging is how a resolution gets orphaned: a gate here once accumulated 26
commits on a detached HEAD with no ref, having already orphaned an earlier merge
resolution. The `reconciliation-merge` LEASE exists for this, and exit **75**
means another runner holds it — defer.

---

## Quick reference

| You see | Branch | Action |
|---|---|---|
| conflict only in lockfiles / folded views / generated docs | 1 | regenerate from the merged tree; declare it in `.mergequeue.yaml` |
| your evidence predates commits now on the base | 2 | `git merge-tree --write-tree`, re-measure, restart |
| clean merge, red gate | 3 | run the gate on the base too; NEW vs pre-existing; unproducible baseline ⇒ refuse |
| real conflict, both sides plausible | 4 | read both sides' **intent**; a fix outranks a feature; prove both tests; second reader |
| real conflict, same intent | 5 | resolve, then run the gate on the merged tree |
| `git branch -d` refuses | — | it is right. Find out why before you reach for `-D` |
