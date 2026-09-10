# Security and publication boundaries

This is research code, not a supported production service. Run archived training scripts only in an isolated environment after reviewing their paths, resource use and dependencies. Do not load untrusted pickle/checkpoint files or relax corpus hash checks.

The publication checker detects selected high-confidence credential/private-key patterns and disallowed tracked file types/paths. It scans the **current tracked snapshot**, not all git history or every possible secret. A clean report is not a security certification. No repository branch-protection, immutable-release or secret-scanning account setting is configured by this code.

The normal verification CI is read-only. The release job has scoped `contents: write` only to create a versioned tag/release and attach the verified compact archive. It runs only on the main branch, never on pull-request code, and refuses to move existing tags. Credentials are supplied through the runner environment, never written into artifacts. External actions are pinned by commit.

For non-sensitive bugs, use a repository issue. **Do not post live secrets or confidential exploit details publicly.** Revoke exposed credentials through their issuer before discussing them. No private advisory channel or response SLA is promised here; no email address has been invented for reporting.
