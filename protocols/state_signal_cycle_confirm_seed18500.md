# LOCKED PROTOCOL — state↔teacher-signal alignment necessity
Date: 2026-08-31 JST
Fresh confirmatory seed: 18500; families 0..4.

This is a NEW claim motivated by the preregistered seed18400 pilot failure mode, not a rescue of the failed residual-equivalence gate.

Intervention and variants are exactly seed18400: L=1; V0=000000, V1=110002, V2=101020, V3=202001, V4=022002, V5=002110; fixed V0–Vj support count 3/6 and TV=.25; A-distance=.5 and entropy matched.

ALIGNED: state x receives its own tokenwise-norm-matched B-head-subspace teacher-delta direction.
CYCLEAVG: changed-state directions are cyclically reassigned by shifts 1,2,3,4,5,0 across successive phase1 steps. Every changed state receives every changed-state direction equally often over 6 steps; the actual state's ALIGNED signal norm is preserved exactly on each step. Unchanged states stay aligned.

For each family and arm define C_j=0.5*[E(0,0)+E(j,j)-E(0,j)-E(j,0)], P_j=-C_j, j=1..5, and M_arm=mean_j(P_j).
Primary D_f=M_CYCLEAVG-M_ALIGNED.

PASS requires ALL, fixed before fresh outcomes:
1) D_f < 0 for 5/5 families (exact one-sided sign p=.03125);
2) pooled mean(M_CYCLEAVG)/mean(M_ALIGNED) <= 0.50;
3) at least 4/5 families have M_CYCLEAVG/M_ALIGNED <= 0.60;
4) M_ALIGNED >= 1.0 B-step in at least 4/5 families (nontrivial matching baseline);
5) audits PASS: A-distance deviation<=1e-8, entropy gap<=1e-7, B-head change<=1e-10, common A-range<=0.01, norm relerr<=1e-5, degeneracy<=1e-6.

No rescue seed, no threshold changes. If PASS, interpret as evidence that persistent state-specific assignment of B-decodable teacher-signal directions is necessary for most source-target matching benefit under this L=1 synthetic protocol. It does NOT by itself prove that this assignment explains the finer equal-count identity residual.
