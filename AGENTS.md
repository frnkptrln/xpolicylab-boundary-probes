# Working in this repository

Independent, falsifiable probes at the boundary between an XPolicyLab policy
adapter and its environment. Deterministic, offline, no learned policy.

## Checks

```bash
python -m pip install -e '.[test]'
python -m pytest
python -m boundary_probes all --output artifacts/latest.json   # exits nonzero if a required control fails
```

## Rules

- Every probe keeps equivalence and sensitivity separate: what must be
  invariant and what must be detectable are different assertions.
- `artifacts/` is ignored; results travel in pull request descriptions.
- Stage 04 (policy divergence) stays "planned" until a policy exists; do not
  simulate one to fill the row.

## Working alongside other agents

Frank works with human contributors and AI agents in this repository, often
at the same time. Agents using any model or provider are welcome. The
repository is their shared channel for coordination.

- Work on your own branch (`<agent>/…`, for example `codex/…` or
  `claude/…`). Open a draft pull request as soon as you start and list the
  files you expect to touch. Before you branch, read the open pull requests
  and keep away from their files. Never push to another agent's branch, and
  never to `main` directly.
- A pull request says what changed, why, what was checked (the commands and
  their results) and what remains unverified. Fix a failing check; do not
  weaken or skip it.
- Attribution is optional. Contributors, including AI agents, may identify
  themselves in a pull request or a `Co-authored-by` commit trailer. Credit
  actual contributions and use only names, model details and attribution
  email addresses you know to be accurate. If no attribution email is known,
  use the pull request description. No fixed agent, model or provider name
  is required.
- No status files, task lists or progress notes in the repository. The pull
  requests and the history are the record.
- Frozen material (below) is not edited in place. It changes only through the
  mechanism this repository defines for it, or not at all.
