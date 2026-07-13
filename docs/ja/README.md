# Rugby Possession Value

ラグビーユニオンの状態価値を **Expected Possession Value（EPV）**、観測された状態遷移の価値変化を **Expected Points Added（EPA）** として推定するオープンソース実装です。

## 修正版の位置づけ

Version 0.2では、価値を同一ポゼッション内だけで閉じず、**ポゼッション交代後も次の得点イベントまで追跡**します。

```text
現在状態
  ↓
同じチームが保持して次状態へ
または
相手ボールの次状態へ
  ↓
次の得点イベントまたはハーフ終了
```

キックやターンオーバーで相手へボールが渡った場合、次状態は相手視点で記録されるため、現在チーム視点では価値を符号反転します。

```text
V(s)
= Σ P_same(s→s') V(s')
+ Σ P_opp(s→s') [-V(s')]
+ Σ P_score(s→o) u(o)
```

## データからの遷移復元

公開データは試合内で `ID` 順に並んでいます。隣接行を比較し、次の状態を復元します。

1. 次行の `Team_In_Poss` から保持継続か保持交代かを判定
2. 次行の `Points_Difference` を現在チーム視点へ変換
3. 得点差が変わっていれば得点イベントで吸収
4. 得点差が変わっていなければ次状態へ遷移

これにより、オープンプレーキック後に相手がどのエリア・起点で保持することが多いかも、粗い状態粒度では推定できます。

## 状態定義

```text
state = (
  possession_origin,
  location_zone,
  lateral_zone,
  phase_bucket
)
```

- `possession_origin`: Lineout / Scrum / Turnover Steal / Kick (Open Play) / Restart Kick / Quick Tap
- `location_zone`: 8つの縦方向エリア
- `lateral_zone`: Left / Centre / Right
- `phase_bucket`: 1 / 2-3 / 4-6 / 7+

## EPVとEPA

### EPV

ある状態から、保持交代を含めて次の得点イベントまたはハーフ終了までに得るネット期待得点です。

### EPA

```text
同じチーム保持:
EPA = EPV(after) - EPV(before)

相手へ保持交代:
EPA = -EPV(after) - EPV(before)

得点イベント:
EPA = reward - EPV(before)
```

## 利用方法

```bash
python -m pip install -e ".[dev]"
rugby-value fetch-data
rugby-value fit data/raw/phase_2018-19.csv --output models/premiership-2018-19
rugby-value table models/premiership-2018-19/model.json --output data/processed
```

## 出力

- `model.json`
- `state_values.csv`
- `start_state_values.csv`
- `transition_probabilities.csv`
- `student_table.csv`
- `observed_epa.csv`

## 重要な制約

公開データには、キャリー、パス、ボックスキック、テリトリーキックなどの個別アクションラベルがありません。そのため本版は、**ポゼッション交代を含む観測状態価値モデル**です。選ばなかったプレーとの反実仮想比較までは行いません。
