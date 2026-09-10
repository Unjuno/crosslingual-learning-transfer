# Third-party data and source provenance

Repository code keeps its existing [Apache-2.0 license](../LICENSE). That license is **not** applied to external corpus text or annotations.

## Executed natural-text corpus

Vim and TeX Live software translation/help strings; exact source/output sizes and SHA256 are in [the frozen manifest](../protocols/natural_ja_en_s2_corpus_manifest_2026-09-06.json). [Preparation code](../experiments/prepare_natural_r1_system_translations.py) is public. Corpus text is not redistributed. The script's presence is not a guarantee that a current system package contains the historical version. Source-package terms govern reuse.

Japanese-sourced text can contain ASCII/Latin fragments. The restriction is corpus provenance, not disjoint byte identities. Exact normalized train/eval overlap and changed document evaluation are documented in [R2 diagnostics](R2_DIAGNOSTICS_2026-09-08.md).

## Unexecuted independent-corpus preparation

The archived UD preparation script pins Japanese-GSD `8e5794fae6d25796437a64a08c497a5c870b7b09`, English-EWT `4c89b5833a70aa5ed3a00bad2f23f57992cc7df8`, and Russian-GSD `27975702bf9507f0a244007f47345239a2d39c2f` (r2.16 references). It does not make an experimental-result claim. The upstream Japanese GSD documentation distinguishes underlying text from annotation rights. Consult each pinned upstream README/license; do not assume one uniform license for text, annotations and this code.

Sources: [Japanese GSD documentation](https://github.com/UniversalDependencies/UD_Japanese-GSD/blob/8e5794fae6d25796437a64a08c497a5c870b7b09/README.md), [English EWT](https://github.com/UniversalDependencies/UD_English-EWT/tree/4c89b5833a70aa5ed3a00bad2f23f57992cc7df8), [Russian GSD](https://github.com/UniversalDependencies/UD_Russian-GSD/tree/27975702bf9507f0a244007f47345239a2d39c2f).

## Publication policy

No corpus documents, generated substitutes for missing raw data, model weights, private credentials or local handoff ZIPs are included in the compact release. Aggregated numerical results, methods, source references and hashes remain public. The publication scanner checks common forbidden paths/extensions; it cannot adjudicate every copyright or privacy issue.
