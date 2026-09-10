# Frozen experimental research record — v1.0.0-research

The current sequence is closed. This is the first versioned **research snapshot**, not a stable product, paper, or human-learning recommendation. No new learning experiment was run for publication.

Read `README.md` / `README.ja.md`, `docs/FINAL_STATUS_2026-09-10.md`, and `docs/REPRODUCIBILITY.md`.

## Retained findings and limitations

- Controlled synthetic A-side interventions can help or harm later B learning; representation/decoder interventions matter in the tested systems.
- Natural R1 weight1.0 remains **registered FAIL** (Japanese phase1 penalty above its one-percent ceiling).
- R2 weight0.5 remains **registered PASS** on its ten-update grid (mean English interval saving 5.1881%, 10/10 favorable; Japanese penalty 0.6293%). Every-update timing gives **4.5692%**, so robust ≥5% is not established.
- Common-head/frozen-body diagnostics retain an endpoint body-origin difference, but reuse existing seeds and add **zero independent trials**. Initial knowledge and learning efficiency are not fully separated.
- An English-aware teacher supplies hidden targets on Japanese-sourced inputs. Target-unaware text-only, independent-corpus, large-LLM, human and total-compute claims are not established.

## Contents and verification

The attached compact ZIP includes public source snapshots, protocols, numerical records, documentation and a file inventory. With Python3.11–3.13, run `python scripts/verify_publication.py` from its root; no pip install, network, GPU or model execution is required. The attached report records the CI checks and commit used to build this package. `SHA256SUMS.txt` identifies the attached files.

**Not included:** third-party corpus text, large raw traces, checkpoints or private/local archives. Their hashes and availability boundaries are documented, but public download locations are not supplied. Verification checks arithmetic and publication integrity, not end-to-end training or the truth of every scientific claim.

Historical scientific files are byte-preserved from evidence base `63a6a056f9575d027dde09d3d19be172e821cd95`. Corrections require a visible note and new version; the workflow refuses to move a published tag. No X/social-media post is sent.
