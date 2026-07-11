# データ

## 取得元

初期実装は `WhartonSABI/rugby-ep` で公開されている `data/phase_2018-19.csv` を利用します。

- 2018/19 English Premiership Rugby
- 132試合
- 35,199フェーズ
- Git blob SHA: `6cc793ef8d21b5ec08dc7a21a37533ad4e7d6e35`

生データは本リポジトリへ再配布せず、`rugby-value fetch-data` で取得します。利用条件は取得元を確認してください。

## 主な列

```text
ID, Round, Home, Away, Phase, Team_In_Poss,
Location, Side, Play_Start, Points_Difference,
Seconds_Remaining, cards, Outcome
```

## 座標近似

`Location`は保持チームの攻撃方向に正規化されています。

| Location | X（自陣トライライン=0） |
|---|---:|
| Goal-5m (own) | 2.5 |
| 5m-22m (own) | 13.5 |
| 22m-10m (own) | 31.0 |
| 10m-Half (own) | 45.0 |
| Half-10m (opp) | 55.0 |
| 10m-22m (opp) | 69.0 |
| 22m-5m (opp) | 86.5 |
| 5m-Goal (opp) | 97.5 |

Yは `Left=11.67`, `Centre=35`, `Right=58.33` とします。
