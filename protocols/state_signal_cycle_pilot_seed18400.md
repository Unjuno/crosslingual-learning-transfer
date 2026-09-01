# LOCKED PROTOCOL — local state↔teacher-signal identity adjudication
Date: 2026-08-31 JST
Fresh pilot seed: 18400; family0 only.

Motivation: fixed-support-count residual survived global stationary/exposure/path scalars, teacher-signal scalar distances, fixed row weights, and L=6→L=1 context removal. At L=1 the remaining candidate is a stable current-state ↔ teacher-signal direction association.

Variants: V0=000000, V1=110002, V2=101020, V3=202001, V4=022002, V5=002110, exactly as seeds18100–18300. Every V0–Vj differs in exactly 3/6 support blocks, so TV(V0,Vj)=0.25 exactly; entropy and TV(A,B)=0.5 are matched.

Sequence length is fixed to L=1 in all arms.

For every source teacher, compute the B-head-subspace, tokenwise-norm-matched teacher delta table for all 12 current states. Compare:
- ALIGNED: current state x receives its own Bsub delta direction delta_Bsub(x).
- CYCLEAVG: on changed states only, delta directions are cyclically reassigned by shifts 1,2,3,4,5,0 across consecutive phase1 steps. Each changed state therefore receives every changed-state Bsub direction equally often over each 6-step cycle. The intervention norm for each actual current state x is rescaled to exactly match the ALIGNED norm at x on every step. Unchanged states remain aligned.

Everything else is identical: same A examples/targets, CE, architecture, phase1 steps, optimizer, calibration, frozen B head, target-learning CRN, and B kernels.

For each arm define C_j=0.5*[E(0,0)+E(j,j)-E(0,j)-E(j,0)], P_j=-C_j, j=1..5. Let R=max(P)-min(P), SD=sd(P).

Pilot gate requires ALL:
1) R_CYCLEAVG <= 0.70 * R_ALIGNED;
2) SD_CYCLEAVG <= 0.70 * SD_ALIGNED;
3) mean(P_CYCLEAVG) >= 0.50 B-step (exclude trivial signal collapse);
4) audits PASS: A-distance deviation<=1e-8, entropy gap<=1e-7, B-head change<=1e-10, common A-range<=0.01, norm relerr<=1e-5, degeneracy<=1e-6.

No rescue seed and no margin tuning on seed18400. If PASS, lock fresh 5-family confirmatory before outcomes. If FAIL, conclude that stable current-state↔Bsub-direction assignment is not the sole source of fixed-count identity dispersion.
