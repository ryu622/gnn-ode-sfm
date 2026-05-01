# 要件定義書：GNN-ODE-SFM

**プロジェクト名**: gnn-ode-sfm  
**研究題目**: イベント同期型トラッキングデータを用いた、状況適応型 Social Force Graph Neural Network によるサッカーの戦術解析  
**更新日**: 2026-04-30

---

## 1. 解くタスクの定義

### 1.1 タスク：多エージェント軌跡予測（Multi-Agent Trajectory Prediction）

| 項目 | 内容 |
|---|---|
| 入力 | 過去 `T` フレーム分の全選手ポジション・速度 + 同期イベントフラグ |
| 出力 | 未来 `K` フレーム先の全選手ポジション（座標） |
| エージェント数 | 最大 22 名（ホーム 11 + アウェイ 11）、ボール除く |
| 損失関数 | MSE（平均二乗誤差） `L = ||x_pred - x_true||^2` |

### 1.2 パラメータ（初期値）

| パラメータ | 値 | 備考 |
|---|---|---|
| `T`（観測ウィンドウ） | 25 フレーム（= 1 秒） | 後で調整可 |
| `K`（予測ホライズン） | 25 フレーム（= 1 秒） | 後で調整可 |
| サンプリング周波数 | 25 Hz | Metrica Sports 準拠 |

---

## 2. 使用データ

### 2.1 データソース

Metrica Sports オープンデータセット（`gnn-counterattack-xai` で使用済みのものと同一フォーマット）

| ファイル | 内容 |
|---|---|
| `Sample_Game_1_RawTrackingData_Home_Team.csv` | ホームチームの全選手トラッキングデータ |
| `Sample_Game_1_RawTrackingData_Away_Team.csv` | アウェイチームの全選手トラッキングデータ |
| `Sample_Game_1_RawEventsData.csv` | イベントデータ（種別・発生フレーム・実行選手） |

### 2.2 生データのカラム構造

```
トラッキングデータ（各チーム）:
  Period | Frame | Time [s] | Player11_x | Player11_y | Player1_x | Player1_y | ... | Ball_x | Ball_y

イベントデータ:
  Type | Period | Frame | From | To | Team | Start Frame | Start Time [s] | End Frame | End Time [s] | ...
```

### 2.3 前処理パイプライン（`src/data/preprocess.py`）

`gnn-counterattack-xai/CounterAttack_Prediction_Processng.ipynb` の前処理を `.py` モジュールとして再実装する。

**ステップ1: 生データ読み込み**
- `pd.read_csv()` でホーム・アウェイ・イベントの3ファイルを読み込む
- ヘッダー行（行0）を除去し、カラム名を正規化

**ステップ2: カラム整形**
- 選手カラムを `home_p{id}_x`, `home_p{id}_y`, `away_p{id}_x`, `away_p{id}_y` 形式に統一
- `Period`, `Frame`, `Time_s` を index として保持

**ステップ3: 欠損値処理**
- 線形補間（`interpolate(method='linear')`）で NaN を埋める
- 出場していない選手のカラムは除去

**ステップ4: 速度ベクトル計算**
- 差分 `v_t = (x_{t+1} - x_{t-1}) / (2 * dt)` で数値微分（中心差分）

**ステップ5: イベントフラグのエンコード**
- イベント種別を以下のカテゴリに集約してワンホット化

| フラグ名 | 対応イベント種別 |
|---|---|
| `event_pass` | PASS |
| `event_shot` | SHOT |
| `event_ball_lost` | BALL LOST |
| `event_ball_start` | BALL START |
| `event_challenge` | CHALLENGE |

- 各フレームに対してアクティブなイベントフラグを付与（イベント継続中のフレームにフラグ = 1）
- ボール保持チームフラグ `home_possession` も付与

**ステップ6: スライディングウィンドウでシーケンス生成**
- `(T + K)` フレーム長のウィンドウをスライドさせてサンプルを生成
- ストライド: 5 フレーム（20% 重複）
- 前後半の境界（Period 変わり目）をまたぐウィンドウは除外

**ステップ7: 正規化**
- 座標: フィールドサイズ（長さ 105m、幅 68m）で除算 → `[-1, 1]` に正規化
- 速度: 訓練セットの標準偏差で除算

**ステップ8: データ分割と保存**
- 訓練 / 検証 / テスト = 70 / 15 / 15 %（シーケンシャル分割、リーク防止）
- `.pt` 形式（`torch.save()`）で保存

---

## 3. システムアーキテクチャ

### 3.1 全体データフロー

```
[入力]
観測シーケンス x_{0..T-1}: (T, N, 4)  ← (x, y, vx, vy) × N 選手
イベントフラグ e_{0..T-1}:  (T, E)     ← E 種類のイベントフラグ

           ↓
┌────────────────────────────────┐
│  1. Context Encoder (MLP)      │  e → embedding c ∈ R^d_c
└────────────────────────────────┘
           ↓
┌────────────────────────────────┐
│  2. GCN Node Feature Encoder   │  (state, c) → node features h_i
└────────────────────────────────┘
           ↓
┌────────────────────────────────┐
│  3. Parameter NN (MLP)         │  (h_i, h_j) → w_ij, σ_ij per edge
└────────────────────────────────┘
           ↓
┌────────────────────────────────┐
│  4. Social Force ODE Layer     │  dx/dt = f_SF(x, w, σ)  ← 物理動力学
│     Neural ODE Solver          │  torchdiffeq.odeint()
└────────────────────────────────┘
           ↓
[出力]
予測シーケンス x̂_{T..T+K-1}: (K, N, 2)  ← 未来座標
```

### 3.2 コンポーネント詳細

#### A. Context Encoder

```
入力: イベントフラグ e ∈ R^E  （各フレームのワンホット集合）
出力: コンテキスト埋め込み c ∈ R^{d_c}

構造: Linear(E, 64) → ReLU → Linear(64, d_c)
d_c = 32
```

#### B. GCN Node Feature Encoder

```
入力: 各選手の状態 s_i = [x_i, y_i, vx_i, vy_i, team_flag] + c ∈ R^{4+1+d_c}
出力: ノード埋め込み h_i ∈ R^{d_h}

構造: GCNConv × 2 層（d_in → d_h → d_h）
d_h = 64
グラフ構造: 距離閾値 r_thresh 以内の選手ペアにエッジを張る（デフォルト: 10m）
            ホーム↔アウェイ間のエッジも含む（全方向）
```

#### C. Parameter NN（物理係数予測）

```
入力: エッジ (i, j) に対し、concat(h_i, h_j) ∈ R^{2*d_h}
出力: w_ij ∈ R^+, σ_ij ∈ R^+  （各エッジの Social Force 係数）

構造: Linear(2*d_h, 64) → ReLU → Linear(64, 2) → Softplus
Softplus で正値を保証
```

#### D. Social Force ODE Layer（核心部）

**物理方程式（斥力モデル: 指数関数型）**

```
f_ij = w_ij * exp(-r_ij / σ_ij) * (x_i - x_j) / r_ij

r_ij = ||x_i - x_j||_2  （選手間ユークリッド距離）
w_ij ∈ R^+              （斥力強度）
σ_ij ∈ R^+              （影響距離スケール）
```

**各選手への合力**

```
F_i = Σ_{j≠i, r_ij < r_thresh} f_ij
```

**Neural ODE の状態方程式**

```
d/dt [x_i, v_i] = [v_i, F_i / m]   （m = 1 と単純化）
```

ここで `F_i` は GNN（Parameter NN）が出力した係数を使って計算する。  
`torchdiffeq.odeint(func, y0, t, method='euler')` で時間積分する。  
微分可能なため、ODE ソルバーを通じて Parameter NN の重みへ誤差逆伝播される。

---

## 4. ファイル構成

```
gnn-ode-sfm/
├── REQUIREMENTS.md          ← このファイル
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml         ← ハイパーパラメータ管理
├── data/
│   └── raw/                 ← 生データ配置場所（.gitignore）
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py    ← 前処理パイプライン
│   │   └── dataset.py       ← PyTorch Dataset / DataLoader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── context_encoder.py
│   │   ├── gcn_encoder.py
│   │   ├── parameter_nn.py
│   │   ├── social_force_ode.py  ← ODEFunc + torchdiffeq 統合
│   │   └── sfm_gnn.py       ← 全体モデルのラッパー
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_debug.ipynb
│   └── 03_visualization.ipynb
└── experiments/
    └── runs/                ← ログ・チェックポイント（.gitignore）
```

---

## 5. ハイパーパラメータ（`config/default.yaml`）

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  obs_len: 25          # T: 観測フレーム数
  pred_len: 25         # K: 予測フレーム数
  stride: 5
  r_thresh: 10.0       # グラフ構築の距離閾値 [m]
  field_length: 105.0
  field_width: 68.0

model:
  d_c: 32              # Context Encoder 出力次元
  d_h: 64              # GCN ノード埋め込み次元
  gcn_layers: 2
  ode_method: "euler"  # torchdiffeq solver method
  ode_rtol: 1e-3
  ode_atol: 1e-4

train:
  batch_size: 32
  lr: 1e-3
  epochs: 100
  weight_decay: 1e-4
  device: "cuda"
```

---

## 6. 評価指標

| 指標 | 計算式 | 意味 |
|---|---|---|
| **ADE** (Average Displacement Error) | `mean(||x̂_t - x_t||)` over all t, N | 全フレーム平均変位誤差 |
| **FDE** (Final Displacement Error) | `mean(||x̂_{T+K} - x_{T+K}||)` over N | 最終フレーム変位誤差 |
| **CR** (Collision Rate) | 選手間距離 < 1m のフレーム割合 | 物理的妥当性（重なり） |
| **RMSE** | `sqrt(mean(||x̂ - x||^2))` | 従来研究との比較用 |

---

## 7. ベースラインモデル

| モデル | 概要 | 実装方針 |
|---|---|---|
| **Constant Velocity** | 等速直線運動で予測 | ルールベース |
| **Vanilla GCN** | GCN のみ、物理制約なし | `src/models/baselines/gcn_baseline.py` |
| **Social Force（固定係数）** | 物理式のみ、GNN 不使用 | `src/models/baselines/sfm_fixed.py` |
| **提案手法（GNN-ODE-SFM）** | 本研究の手法 | `src/models/sfm_gnn.py` |

---

## 8. 実装フェーズ（スケジュール）

### フェーズ 1：物理エンジン層の構築（2026年5月）
- [ ] `social_force_ode.py`: `ODEFunc` の実装と autograd テスト
- [ ] `torchdiffeq` が微分可能であることをユニットテストで確認
- [ ] トイデータ（sin 波軌跡）での動作検証

### フェーズ 2：GNN 統合とデータパイプライン（2026年6月）
- [ ] `preprocess.py`: Metrica データ前処理の実装
- [ ] `dataset.py`: スライディングウィンドウ Dataset の実装
- [ ] `gcn_encoder.py` + `parameter_nn.py` の実装
- [ ] 全体モデル `sfm_gnn.py` の結合テスト

### フェーズ 3：訓練・評価・解析（2026年7〜8月）
- [ ] `train.py` の実装（early stopping, チェックポイント保存）
- [ ] `evaluate.py`: ADE/FDE/CR/RMSE の計算
- [ ] `visualize.py`: 係数 `w_ij` の時系列可視化（ポテンシャル場）
- [ ] ベースライン比較実験

---

## 9. 未解決事項・今後の検討点

| 項目 | 内容 |
|---|---|
| 予測ホライズン | T=K=25（1秒）は仮置き。精度と計算コストのトレードオフで調整 |
| ODE ソルバー | Euler 法から始め、精度改善が必要なら `dopri5`（RK4-5）へ切り替え |
| 選手個性の埋め込み | フェーズ 3 以降に Player ID Embedding を追加検討 |
| ボール軌跡 | 初期は除外。物理モデルが安定してから追加を検討 |
| マルチゲームへの拡張 | まず Sample_Game_1 で動作確認後、複数試合に拡張 |
