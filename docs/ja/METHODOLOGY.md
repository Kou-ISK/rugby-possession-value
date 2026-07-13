# 方法論

## 1. 観測単位

公開データの各行は1フェーズです。試合内では `ID` 順に並んでおり、`Team_In_Poss`、`Points_Difference`、`Play_Start`、`Location` を使って次状態への遷移を復元します。

従来版のようにポゼッションごとに系列を切るのではなく、**試合時系列全体を次の得点イベントまで接続**します。

## 2. 状態空間

```text
s = (origin, location, side, phase_bucket)
```

- `origin`: ポゼッション起点
- `location`: 縦方向の8エリア
- `side`: Left / Centre / Right
- `phase_bucket`: 1 / 2-3 / 4-6 / 7+

## 3. 得点イベントの検出

`Points_Difference` は、その行で保持しているチーム視点です。現在行を `t`、次行を `t+1` とすると、次行の得点差を現在チーム視点へ変換します。

```text
保持継続:
next_pd_from_source = PD(t+1)

保持交代:
next_pd_from_source = -PD(t+1)
```

そして、

```text
score_delta = next_pd_from_source - PD(t)
```

が0でなければ、現在行と次行の間で得点が起きたと判定し、現在行の `Outcome` へ吸収させます。0なら次状態へ遷移します。

## 4. 遷移の分類

得点なし遷移を2種類に分けます。

```text
P_same[i,j]
状態i → 自チーム保持の状態j

P_opp[i,j]
状態i → 相手保持の状態j

P_absorb[i,o]
状態i → 得点結果o
```

状態ごとの標本数が少ないため、同じ起点・フェーズ帯および全体分布へのEmpirical Bayes平滑化を行います。

## 5. 符号付きマルコフ報酬過程

相手保持の次状態 `s'` は相手視点で定義されています。現在チーム視点では価値が反転するため、Bellman方程式は次です。

```text
V = P_same V - P_opp V + P_absorb u
```

したがって、

```text
(I - P_same + P_opp) V = P_absorb u
```

を満たします。

## 6. 最終結果確率

保持が相手へ移る場合は、`For Try` と `Against Try`、`For Penalty Kick` と `Against Penalty Kick` のように結果ラベルも反転します。

各アウトカムについて、保持交代時のラベル反転を含むブロック線形方程式を解き、次得点イベントの確率分布を求めます。

## 7. EPA

```text
保持継続:
EPA = V(after) - V(before)

保持交代:
EPA = -V(after) - V(before)

得点:
EPA = reward - V(before)
```

## 8. 検証

- `P_same + P_opp + P_absorb` の行和が1
- 吸収結果確率の行和が1
- 符号反転込みBellman残差が許容誤差内
- モデルJSONの保存・復元
- 保持交代と得点差変化を含むfixtureテスト

## 9. 解釈上の制約

本モデルは、公開データから推定した**観測方針下の状態価値**です。ポゼッション交代後まで価値を追跡できますが、キャリー、パス、ボックスキック、テリトリーキックなどのアクションラベルがないため、個別アクション間の因果的・反実仮想的な優劣は推定しません。
