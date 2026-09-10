# Publication controls and remaining limitations

## Gates implemented for v1.0.0-research

| Gate | What is checked | What is not promised |
|---|---|---|
| Final status | Frozen scope, R1 FAIL, R2 registered-grid PASS and adverse dense timing retained | All research questions solved |
| Scientific snapshot | Existing protocol/result/experiment source bytes match the evidence-base manifest | Cryptographic proof of original execution time |
| Record arithmetic | Five existing verifier families and positive/negative/malformed-record tests | New model training or raw-checkpoint reproduction |
| Public navigation | Local inline Markdown file links, English/Japanese entry pages and current evidence map | External link permanence or every heading fragment |
| File exposure | Manifest-listed tracked paths/types/size, selected secret patterns, no temporary sandbox URLs | Comprehensive security, copyright or whole-history scan |
| Package | Only inventoried text files plus source-commit marker; ZIP CRC and SHA256 | Third-party corpus or full raw/checkpoint availability |
| Release | Full verification before tag/release; trusted main only; no tag moves or release overwrite | Repository-wide immutable-release/branch-protection configuration |
| Dependencies | Verification uses standard library; external CI actions pinned by SHA; read-only checks | A full historical OS/wheel/ABI lock |

Run `python scripts/verify_publication.py` for the actual status and report. The workflow validates Python 3.11 and 3.13. `scripts/build_release.py` refuses to package after a failed gate. Release metadata is versioned; tags are not moved by the publisher.

## Deliberate remaining boundaries

The code/results were already in a public repository. This cleanup does not change repository visibility, rewrite history, delete branches or archive the repository. It freezes the **research scope**, not the ability to correct errors.

Most large historical artifacts remain outside the compact public release. [Availability catalog](../publication/artifact_catalog.json) has no invented public download links. Missing state-signal helpers remain blocked; independent-corpus training was not run. No new experiment, peer review, paper, DOI or social-media post is created by packaging.

For a reviewed documentation or software change: stage only intended files, run `python scripts/refresh_inventory.py`, stage the refreshed manifest, then run the full verifier. A scientific correction requires a separate visible correction note, amended provenance and a new version; do not disable snapshot guards to hide it.
