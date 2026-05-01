# フェーズ2 引継ぎノート

**作成日**: 2026-04-30  
**フェーズ1完了状態**: `tests/test_phase1.py` 11/11 PASS・検証スクリプト全シナリオOK

---

## 1. フェーズ1で確立したこと

| 項目 | 内容 |
|---|---|
| `SocialForceFunc` | `torchdiffeq` の ODE 動力学関数。`forward(t, y)` で `dy/dt` を返す |
| `SocialForceODELayer` | `odeint` ラッパー。`y0` + 時間点列 `t` を受け取り軌跡を返す |
| autograd疎通 | `odeint` を通じて `log_w`・`log_sigma`・外部テンソルへの逆伝播を確認済み |
| バッチ対応 | `y0: (B, N, 4)` でも `(N, 4)` でも動作する（`unsqueeze(-2/-3)` 設計） |
| Phase2インタフェース | `set_edge_coefficients(w, sigma)` を呼ぶだけで内部スカラーをGNN出力に差し替え可能 |

---

## 2. フェーズ2で実装するもの

REQUIREMENTS.md フェーズ2の内容：

```
- preprocess.py   : Metrica データ前処理の実装
- dataset.py      : スライディングウィンドウ Dataset の実装
- context_encoder.py  : イベントフラグ → 埋め込みベクトル
- gcn_encoder.py      : GCNConv でノード埋め込みを計算
- parameter_nn.py     : エッジごとの (w_ij, σ_ij) を予測
- sfm_gnn.py          : 上記を結合した全体モデルラッパー
```

---

## 3. GNN → SocialForceFunc の接続方法（最重要）

フェーズ1の `SocialForceODELayer.forward()` のシグネチャ：

```python
def forward(
    self,
    y0: Tensor,              # (..., N, 4)  初期状態
    t: Tensor,               # (K+1,)       時間点列
    w: Optional[Tensor],     # (N, N) or None
    sigma: Optional[Tensor], # (N, N) or None
) -> Tensor:                 # (K+1, ..., N, 4)
```

フェーズ2のモデル (`sfm_gnn.py`) は以下の流れで呼び出す：

```python
# 1. Context Encoder でイベント埋め込みを計算
context = self.context_encoder(event_flags)          # (B, d_c)

# 2. GCN でノード埋め込みを計算
node_feats = self.gcn_encoder(state, edge_index, context)  # (B, N, d_h)

# 3. Parameter NN でエッジごとの係数を予測
w, sigma = self.parameter_nn(node_feats, edge_index)  # (B, N, N) 各

# 4. ODE レイヤーに渡して積分（フェーズ1コードをそのまま使う）
y_pred = self.ode_layer(y0, t, w=w, sigma=sigma)     # (K+1, B, N, 4)
```

`w` と `sigma` が `(B, N, N)` になる点に注意（バッチ次元が追加される）。  
`SocialForceFunc._social_force()` 内の `force_mag = w * torch.exp(-safe_dist / sigma)` で  
`(B, N, N)` × `(B, N, N)` としてブロードキャストされるため、コードの修正は不要。

---

## 4. グラフ構築（`edge_index` の作り方）

距離閾値 `r_thresh`（デフォルト 10.0 m）以内の選手ペアにエッジを張る。

```python
import torch
from torch_geometric.utils import dense_to_sparse

def build_edge_index(pos: torch.Tensor, r_thresh: float) -> torch.Tensor:
    """
    pos: (N, 2)  選手座標
    returns: edge_index (2, E)
    """
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # (N, N, 2)
    dist = diff.norm(dim=-1)                      # (N, N)
    N = pos.shape[0]
    eye = torch.eye(N, dtype=torch.bool, device=pos.device)
    adj = (dist < r_thresh) & ~eye               # (N, N) bool
    edge_index, _ = dense_to_sparse(adj.float())
    return edge_index                             # (2, E)
```

バッチ処理時は `torch_geometric.data.Batch` を使うか、バッチ内で固定グラフ構造を仮定する（全選手が常に同じ接続を持つ場合）。

---

## 5. Parameter NN の設計

エッジ `(i, j)` ごとに `w_ij` と `sigma_ij` を出力する MLP：

```python
class ParameterNN(nn.Module):
    def __init__(self, d_h: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_h, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softplus(),   # 正値保証（w, sigma > 0）
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor):
        # h: (N, d_h)  ノード埋め込み
        # edge_index: (2, E)
        src, dst = edge_index
        edge_feats = torch.cat([h[src], h[dst]], dim=-1)  # (E, 2*d_h)
        out = self.mlp(edge_feats)                         # (E, 2)
        w_edge = out[:, 0]                                 # (E,)
        sigma_edge = out[:, 1]                             # (E,)

        # SocialForceFunc が (N, N) を期待するため密行列に変換
        N = h.shape[0]
        w_dense = torch.zeros(N, N, device=h.device)
        sigma_dense = torch.ones(N, N, device=h.device)   # default=1 (no interaction)
        w_dense[src, dst] = w_edge
        sigma_dense[src, dst] = sigma_edge
        return w_dense, sigma_dense                        # (N, N), (N, N)
```

---

## 6. データパイプラインの注意点

### 座標の正規化と `r_thresh` のスケール合わせ

`preprocess.py` でフィールドサイズ正規化（÷105, ÷68）を行う場合、  
`r_thresh` も同様にスケール変換が必要：

```python
# 10 m → 正規化後スケール
r_thresh_x = 10.0 / 105.0  # ≈ 0.095
r_thresh_y = 10.0 / 68.0   # ≈ 0.147
# → 等方的に扱う場合は r_thresh = 10.0 / 105.0 など統一する
```

**推奨**: フェーズ2実装開始時に「正規化あり or なし」を決定し、  
`config/default.yaml` の `r_thresh` を対応する値に更新する。

### イベントフラグのフレーム同期

Metrica のイベントデータは `Start Frame` / `End Frame` を持つ。  
イベント継続中のフレームに `1` を立てる処理は `preprocess.py` に集約する  
（`gnn-counterattack-xai/CounterAttack_Prediction_Processng.ipynb` の実装を参考）。

---

## 7. フェーズ2の実装順序（推奨）

1. **`preprocess.py`** を先に完成させ、`data/processed/` に `.pt` ファイルを生成する
2. **`dataset.py`** でスライディングウィンドウ Dataset を実装し、DataLoader の動作を確認
3. **`context_encoder.py`** → **`gcn_encoder.py`** → **`parameter_nn.py`** の順に単体テスト
4. **`sfm_gnn.py`** で全体を結合し、ダミーデータでの forward パスを確認
5. `train.py` でループを実装、過学習確認（少量データで loss が下がるか）

---

## 8. フェーズ1で判明した制約・留意事項

| 項目 | 内容 |
|---|---|
| ODE ソルバー | Euler 法は速いが精度が低い。精度が不足した場合は `method="rk4"` or `"dopri5"` へ変更。ただし計算コスト増。 |
| w と sigma の非同定性 | Scenario B で確認。トラジェクトリの L2 損失だけでは (w, σ) の一意な回復が困難な場合がある。正則化や初期値の工夫が必要になる可能性あり。 |
| `r_thresh` 外はゼロ勾配 | マスクが `float()` 変換で微分不連続になる。学習中に全エッジが閾値外になる状況は避ける（初期化・データスケールに注意）。 |
| バッチ処理時の `eye` マスク | `SocialForceFunc._social_force()` 内の `torch.eye(N)` は `(N, N)` で固定。`(B, N, N)` の `dist` とブロードキャストされる設計になっているため変更不要。 |
