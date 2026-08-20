# Identity probe adapter

This directory supplies the policy-specific files that differ from
XPolicyLab's `policy/demo_policy` reference.

1. Install this repository in the policy environment.
2. Copy XPolicyLab's `policy/demo_policy` directory to
   `policy/identity_probe`.
3. Replace its `model.py` and `deploy.yml` with the files here.
4. Run the official debug evaluation in plain-array and encoded-observation
   modes before using any simulator-backed client.

The model never decodes runtime images. The XPolicyLab policy server owns that
boundary and supplies decoded RGB arrays to `update_obs` and
`update_obs_batch`.
