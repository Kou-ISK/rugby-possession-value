# Rugby Possession Value

ラグビーユニオンの状態価値を **Expected Possession Value（EPV）**、観測されたプレーの価値変化を **Expected Points Added（EPA）** として推定するオープンソース実装です。

## 学術的な位置づけ

本モデルは、ポゼッションを状態列として扱う**吸収マルコフ報酬過程**です。

```text
状態 s
  ↓ 一時的な状態遷移
次状態 s'
  ↓
トライ、PG、DG、被得点、終了などの吸収結果
```

状態価値は次で計算します。

```text
V = (I - Q)^(-1) B u
```

- `Q`: 一時状態から一時状態への遷移確率
- `B`: 一時状態から得点結果への遷移確率
- `u`: 得点結果のネット得点

## 状態定義

初期モデルの状態は次です。

```text
state = (
  possession_origin,
  location_zone,
  lateral_zone,
  phase_bucket
)
```

- `possession_origin`: Lineout / Scrum / Turnover Steal / Kick Reception相当 / Restart / Quick Tap
- `location_zone`: 8つの縦方向エリア
- `lateral_zone`: Left / Centre / Right
- `phase_bucket`: 1 / 2-3 / 4-6 / 7+

座標は公開データがゾーン形式のため、各ゾーンの代表点へ近似します。

## EPVとEPA

### EPV

ある状態から、当該ポゼッションが終了するまでのネット得点期待値です。

```text
EPV(s) = E[terminal net points | state=s, observed policy]
```

これは「過去のチームが実際に選択してきた戦術方針下」の価値です。

### EPA

観測された状態遷移による価値変化です。

```text
継続遷移: EPA = EPV(after) - EPV(before)
吸収遷移: EPA = terminal_reward - EPV(before)
```

### 判断評価との違い

フェーズデータには、選ばなかったプレーの結果がありません。そのため初期モデルから「パスよりキックが何点優れていた」といった反実仮想は出しません。

選択肢別評価には、少なくとも次が必要です。

```text
action_type
start_x, start_y
end_x, end_y
possession_after
outcome
player_id
```

## 統計処理

- 吸収確率: Beta-Binomial empirical Bayes
- 継続先分布: Dirichlet-Multinomial empirical Bayes
- 得点結果分布: Dirichlet-Multinomial empirical Bayes
- 不確実性: 試合単位のクラスターブートストラップ
- 長いポゼッションの過大評価防止: 得点結果は最終フェーズからのみ吸収遷移として計上

## 利用方法

```bash
python -m pip install -e ".[dev]"
rugby-value fetch-data
rugby-value fit data/raw/phase_2018-19.csv --output models/premiership-2018-19 --bootstrap 500
rugby-value table models/premiership-2018-19/model.json --output data/processed
```

## 出力

- `model.json`: アプリ埋め込み可能なモデル
- `state_values.csv`: 全状態のEPVと吸収確率
- `start_state_values.csv`: エリア×ポゼッション起点の開始時EPV
- `transition_probabilities.csv`: 状態遷移確率
- `student_table.csv`: 学生向けに列を絞った表

## 重要な制約

公開データは正確なX/Y座標やアクション種別を持ちません。したがって、本版は**座標近似による観測方針EPVモデル**であり、最適意思決定モデルではありません。
