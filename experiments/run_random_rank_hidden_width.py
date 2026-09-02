import os
from pathlib import Path
import pandas as pd

import random_rank_state_alignment_gru as core


def main():
    width = int(os.environ.get('HIDDEN_WIDTH', '64'))
    seed = int(os.environ.get('SEED', '5400'))
    family = int(os.environ.get('FAMILY_ONLY', '0'))
    outdir = Path(os.environ.get('OUTDIR', '/mnt/data/random_rank_hidden_width_gru'))
    outdir.mkdir(parents=True, exist_ok=True)

    # Reuse the original experiment implementation exactly; only mutate the
    # learner hidden width, seed, and output root before model construction.
    core.H = width
    core.SEED = seed
    core.ROOT = outdir
    os.environ['SEED_ONLY'] = str(seed)
    os.environ['FAMILY_ONLY'] = str(family)

    core.main()

    curve = outdir / f'family{family}_seed{seed}.csv'
    audit = outdir / f'audit_family{family}_seed{seed}.csv'
    curve_out = outdir / f'family{family}_seed{seed}_H{width}.csv'
    audit_out = outdir / f'audit_family{family}_seed{seed}_H{width}.csv'

    c = pd.read_csv(curve)
    c.insert(2, 'hidden_width', width)
    c.to_csv(curve_out, index=False)
    a = pd.read_csv(audit)
    a.insert(2, 'hidden_width', width)
    a.to_csv(audit_out, index=False)
    curve.unlink()
    audit.unlink()

    print(f'saved family={family} seed={seed} hidden_width={width}', flush=True)


if __name__ == '__main__':
    main()
