#!/usr/bin/env python3
import argparse
import ast
import gettext
import hashlib
import json
import re
from pathlib import Path

LANGS = ["ja", "ru"]


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_po(path):
    entries = []
    cur = None
    mode = None

    def flush():
        nonlocal cur
        if cur and cur.get("msgid") and cur.get("msgstr"):
            entries.append((cur["msgid"], cur["msgstr"]))
        cur = None

    for raw in path.read_text("utf-8").splitlines():
        s = raw.strip()
        if s.startswith("msgid "):
            flush()
            cur = {"msgid": ast.literal_eval(s[6:]), "msgstr": ""}
            mode = "msgid"
        elif cur is not None and s.startswith("msgstr "):
            cur["msgstr"] = ast.literal_eval(s[7:])
            mode = "msgstr"
        elif cur is not None and s.startswith('"'):
            try:
                cur[mode] += ast.literal_eval(s)
            except Exception:
                pass
        elif cur is not None and s == "" and cur.get("msgstr"):
            flush()
            mode = None
    flush()
    return dict(entries)


def load_mo(path):
    with path.open("rb") as f:
        trans = gettext.GNUTranslations(f)
    return {
        k: v
        for k, v in trans._catalog.items()
        if isinstance(k, str) and k and isinstance(v, str) and v
    }


def norm(s):
    s = s.replace("\x00", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\d+", "0", s)


def write_text(path, messages):
    text = "\n".join(x for x in messages if x) + "\n"
    path.write_text(text, encoding="utf-8")
    return {
        "bytes": len(text.encode("utf-8")),
        "lines": sum(1 for x in messages if x),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument(
        "--tex-root",
        type=Path,
        default=Path("/usr/share/texlive/tlpkg/translations"),
    )
    ap.add_argument(
        "--vim-root",
        type=Path,
        default=Path("/usr/share/vim/vim91/lang"),
    )
    args = ap.parse_args()

    tex_paths = {lang: args.tex_root / f"{lang}.po" for lang in LANGS}
    vim_paths = {
        lang: args.vim_root / lang / "LC_MESSAGES" / "vim.mo" for lang in LANGS
    }
    for path in [*tex_paths.values(), *vim_paths.values()]:
        if not path.exists():
            raise SystemExit(f"missing source file: {path}")

    po = {lang: parse_po(path) for lang, path in tex_paths.items()}
    mo = {lang: load_mo(path) for lang, path in vim_paths.items()}
    common_tex = sorted(set.intersection(*[set(po[lang]) for lang in LANGS]))
    common_vim = sorted(set.intersection(*[set(mo[lang]) for lang in LANGS]))

    # Same content-key split convention as the earlier natural-language pilot:
    # Vim msgid hash modulo 10 == 0 is evaluation; all other Vim messages train.
    vim_train = [
        key
        for key in common_vim
        if int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 10 != 0
    ]
    train_set = set(vim_train)
    vim_eval = [key for key in common_vim if key not in train_set]

    args.outdir.mkdir(parents=True, exist_ok=True)
    output = {}
    for lang in ["en", "ja", "ru"]:
        tex_messages = [
            norm(key if lang == "en" else po[lang][key]) for key in common_tex
        ]
        train_messages = tex_messages + [
            norm(key if lang == "en" else mo[lang][key]) for key in vim_train
        ]
        eval_messages = [
            norm(key if lang == "en" else mo[lang][key]) for key in vim_eval
        ]
        output[f"{lang}_train"] = write_text(
            args.outdir / f"{lang}_train.txt", train_messages
        )
        output[f"{lang}_eval"] = write_text(
            args.outdir / f"{lang}_eval.txt", eval_messages
        )

    minimum_train = 80_000
    minimum_eval = 8_000
    for name, meta in output.items():
        need = minimum_train if name.endswith("train") else minimum_eval
        if meta["bytes"] < need:
            raise SystemExit(f"{name} too small: {meta['bytes']} < {need}")

    manifest = {
        "dataset_label": "system_translation_strings_vim_texlive_ja_en_ru",
        "claim_boundary": "Domain-limited software UI/help translation strings; not a broad natural-language corpus.",
        "sources": {
            str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in [*tex_paths.values(), *vim_paths.values()]
        },
        "parallel_key_counts": {
            "tex_common_ja_ru": len(common_tex),
            "vim_common_ja_ru": len(common_vim),
            "vim_train": len(vim_train),
            "vim_eval": len(vim_eval),
        },
        "split_rule": "Vim msgid md5 first32bits % 10 == 0 -> eval; other Vim + all common TeX -> train.",
        "normalization": "NUL/newline to space; collapse whitespace; decimal digit runs -> 0.",
        "outputs": output,
        "minimum_bytes": {"train": minimum_train, "eval": minimum_eval},
    }
    (args.outdir / "CORPUS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
