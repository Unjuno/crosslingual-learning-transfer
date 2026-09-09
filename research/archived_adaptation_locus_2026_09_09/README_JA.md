# 保存済みStudentの更新場所・出力層入替診断

全80適応枝、20,000更新を実行済み。新しい独立seedは0。係数0.5確認試験の既使用32000〜32009を用いた、ローカル事前固定の追加診断であり、別コーパスでの追試ではありません。

詳しい結果と制約は `docs/EXECUTION_REPORT_JA.md`、元診断の方法は `docs/EXPERIMENT_METHOD_JA.md`、交絡に気づいて追加した入替対照は `docs/HEAD_SWAP_METHOD_JA.md` を参照。

## 重要な結果

共通の初期出力層に揃えても、英語教師由来の固定本体は250更新後の英語NLLで10/10の優位。本体出所の平均効果は−0.020383 nats/byte。初期出力層の平均効果は−0.000018で、記述的区間はゼロをまたぎます。初期headの差だけでは終点の優位を説明できません。

ただし英語教師Studentは初めから有利です。損失減少量の増大や純粋な学習率改善は示していません。本体を固定し日本語用headを保存すれば日本語出力は不変ですが、これは構成上の恒等性で、英語の終点性能を犠牲にします。

## 公開記録

コミット対象は方法、固定計画、集計結果、標準ライブラリだけで動く公開検証器です。完全な学習raw traceとcheckpointは実行アーカイブ側に保持し、このコンパクトなGitHubスナップショットには再配布しません。コーパス本文も再配布しません。

```bash
python experiments/verify_records.py
python -m unittest discover -s experiments -p 'test_public_records.py' -v
```

上記は科学的効果の独立再測定ではなく、コミット済み記録のcoverage・算術・有限値・主要判定を検査します。

PyTorchによる元実行・全33テストはPython3.13.5、torch2.10.0+cpu、numpy2.3.5、pandas2.2.3で検証しました。CPUはIntel Xeon Platinum8573C。学習はfloat32、1thread/job、deterministic指定。別CPUとの完全なbitwise一致を保証しません。

## 独立コーパスの準備

`prepare_external_ud.py` はUDの固定commitからtextメタデータのみを抽出し、完全一致・正規化一致・5文字shingle Jaccardによる近重複を監査する次段階用コードです。今回実データの取得とUD学習は未実行。ライセンスは注釈と原文で権利範囲が異なる場合があり、このコードのApache-2.0をデータに適用しません。

## GitHub状態

実験実行時の基準mainは `97d8f8269af7b92e50ad647949f0f20b3cd10612`。実験中のローカル固定は公開事前登録ではないため、結果は **archived / exploratory diagnostic** として扱います。このディレクトリは実行後に公開リポジトリへ統合された監査用スナップショットです。
