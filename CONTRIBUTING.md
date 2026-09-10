# Contributing to a frozen research snapshot

The current sequence is **frozen as of 2026-09-10**. Acceptable maintenance includes reproducibility fixes, arithmetic/error reports, broken documentation links and clearer scope. New outcome-seeking studies require a separate scope and prospective protocol; do not append rescue seeds or tune thresholds on completed cohorts.

Before submitting a change, run `python scripts/verify_publication.py`. Include the release tag/commit, exact command, Python/OS/backend and the smallest non-sensitive error record. Do not attach corpora, model weights, access tokens, private paths or personal data to public issues.

Never silently overwrite a locked protocol or historical result. A genuine correction needs an explicit correction note, provenance for corrected files, affected claims and a new release version. Existing snapshot hash guards intentionally reject changes to scientific records. A maintainer-reviewed correction must deliberately update those guards and document why; a passing test is not a reason to rewrite history.

Release tags must not be moved to make a correction invisible. Research remains frozen while software/documentation corrections are allowed. No external contributions, review or endorsement are implied by CI success.
