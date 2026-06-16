# COG / SB Visualization System の動作原理

## 目的

本システムは、画像・動画・カメラ映像から人体姿勢を推定し、次の指標をフレームごとに算出して可視化する。

- COG: Center of Gravity、全身重心
- BOS: Base of Support、支持基底面
- COP: Center of Pressure、圧力中心の近似

処理結果は `outputs/<session>/frames.jsonl` に保存され、GUI または `cogsb visualize` で入力画像へ重畳表示できる。

## 全体フロー

実装上の主な処理順序は `src/cogsb/pipeline/engine.py` の `AnalysisEngine.run` に集約されている。

1. 入力ソースからフレームを取得する。
2. MediaPipe Pose で 2D / world ランドマークを推定する。
3. 骨格長と時系列拘束を使って 3D 関節位置を補正する。
4. 補正後の関節列を指数移動平均で平滑化する。
5. 身体セグメントごとの質量比から COG を計算する。
6. 足部ランドマークから支持平面と BOS 多角形を計算する。
7. COG の速度・加速度から COP を近似する。
8. COG / BOS / COP と補助情報を `frames.jsonl` に出力する。

## 1. 姿勢推定

姿勢推定は `src/cogsb/pose/mediapipe_estimator.py` の `MediaPipePoseEstimator` が担当する。

入力画像を RGB に変換し、MediaPipe Pose から 33 点のランドマークを取得する。利用可能な場合は `pose_world_landmarks` を優先し、3D 世界座標として扱う。利用できない場合は、正規化 2D 座標を画像サイズでスケールした値を擬似的な 3D 座標として使う。

各ランドマークは次の形で保持される。

```text
p_i = (x_i, y_i, z_i, visibility_i)
```

ここで `i` は MediaPipe のランドマーク番号である。

## 2. 骨格再構成

骨格再構成は `src/cogsb/reconstruction/skeleton_optimizer.py` の `SkeletonReconstructor` が担当する。

MediaPipe の推定値はフレームごとに揺れやすいため、次の残差を同時に小さくする最小二乗問題として関節位置を補正する。SciPy が利用できる場合は `scipy.optimize.least_squares` を使う。

最適化対象を、フレーム `t` の全関節位置

```text
P_t = {p_1, p_2, ..., p_n}, p_i in R^3
```

とする。実装の残差は概念的には次の和である。

```text
E(P_t) =
  w_obs * ||P_t - P_obs||^2
  + w_len * Σ_s (||b_s(P_t) - a_s(P_t)|| - L_s)^2
  + w_smooth * ||P_t - P_{t-1}||^2
  + w_vel * ||P_t - (P_{t-1} + V_{t-1} * dt)||^2
```

各項の意味は次のとおり。

- `P_obs`: MediaPipe が出力した観測関節位置
- `w_obs`: 観測値への追従重み。実装では `damping` と `visibility` を使う
- `s`: 身体セグメント
- `a_s`, `b_s`: セグメント `s` の始点・終点
- `L_s`: 初期フレームまたは信頼できるフレームから推定したセグメント長
- `P_{t-1}`: 前フレームの関節位置
- `V_{t-1}`: 前フレームで推定した関節速度
- `dt`: フレーム間隔

これにより、観測値から大きく外れず、身体セグメント長を保ち、時間的にも急に飛ばない 3D 関節列を得る。

## 3. 時系列平滑化

再構成後の関節列は `Smoother` により指数移動平均で平滑化される。

```text
S_t = alpha * P_t + (1 - alpha) * S_{t-1}
```

現在の実装では `AnalysisEngine` 内で `alpha = 0.30` が使われる。初回フレームでは観測値をそのまま初期値にする。

## 4. COG の計算

COG は `src/cogsb/kinematics/cog.py` の `COGEstimator` が担当する。身体を複数の剛体セグメントに分け、各セグメントの重心を質量比で加重平均する。

セグメント定義は `src/cogsb/reconstruction/body_segments.py` の `SEGMENT_TABLE` にあり、頭部、体幹、上腕、前腕、手、大腿、下腿、足部などを含む。

セグメント `s` の始点を `A_s`、終点を `B_s`、セグメント内の重心位置係数を `alpha_s` とすると、セグメント重心 `C_s` は次で計算する。

```text
C_s = A_s + alpha_s * (B_s - A_s)
```

全身重心 `COG` は、各セグメントの質量比 `m_s` を使って次のように求める。

```text
COG = Σ_s (m_s * C_s) / Σ_s m_s
```

`m_s` は `SEGMENT_TABLE` の質量値を合計値で割った相対質量である。実装では必要に応じて `update_mass_profile` でセグメントごとの補正係数を掛けられる。

COG の速度と加速度は、フレーム間差分から求める。

```text
v_t = (COG_t - COG_{t-1}) / dt
a_t = (v_t - v_{t-1}) / dt
```

初回フレームや `dt` が不正な場合はゼロとして扱う。

## 5. BOS の計算

BOS は `src/cogsb/kinematics/bos.py` の `compute_bos` が担当する。基本方針は、足首・踵・足先のランドマークから支持平面を推定し、その平面上に足部点と COG を射影して、支持基底面の内外判定を行うことである。

### 5.1 支持平面の推定

左右の足首、踵、足先の 6 点を使う。

```text
left ankle  = 27
right ankle = 28
left heel   = 29
right heel  = 30
left foot   = 31
right foot  = 32
```

これらの点集合を `Q = {q_i}` とし、重心を原点候補 `O` とする。

```text
O = mean(Q)
```

中心化した点群 `q_i - O` に SVD を適用し、最小分散方向を支持平面の法線 `up` とする。最大分散方向を平面上の `x_axis` とし、外積で `y_axis` を作る。

```text
up     = smallest_variance_direction(Q)
x_axis = dominant_in_plane_direction(Q)
y_axis = up x x_axis
```

前フレームの支持平面がある場合は、法線や軸の向きをそろえたうえでブレンドし、平面推定の揺れを抑える。

### 5.2 支持平面への射影

任意の 3D 点 `p` は、支持平面座標系では次の 2D 点に変換される。

```text
r = p - O
x = dot(r, x_axis)
y = dot(r, y_axis)
p_support = (x, y)
```

BOS の内外判定では、元の 3D 座標ではなくこの支持平面座標を使う。

### 5.3 接地信頼度

左右の接地信頼度は、踵の高さと足首速度から近似する。

踵が支持平面に近いほど接地しているとみなし、前フレームがある場合は足首速度が小さいほど接地らしいとみなす。実装上は概念的に次の形で合成される。

```text
contact = 0.7 * height_score + 0.3 * low_speed_score
```

`height_score` と `low_speed_score` は `_norm` により 0 から 1 にクリップされた値である。接地信頼度が `0.05` 以上の足部点を BOS 候補に使う。

### 5.4 BOS 多角形

有効な足部点を支持平面へ射影し、その凸包を BOS 多角形とする。Shapely が利用可能な場合は `MultiPoint(...).convex_hull` を使い、利用できない場合は monotonic chain による凸包計算にフォールバックする。

多角形面積は shoelace formula で計算する。

```text
area = 1/2 * |Σ_i (x_i * y_{i+1} - x_{i+1} * y_i)|
```

前フレームの多角形がある場合は、頂点数や移動量が妥当な範囲でブレンドして急激な形状変化を抑える。

### 5.5 COG の内外判定と安定余裕

COG を支持平面へ射影した点を `c = (x, y)` とし、BOS 多角形 `B` に含まれるかを判定する。

Shapely が利用可能な場合は `Polygon.contains` / `touches` と境界距離を使う。利用できない場合は ray casting による点-in-多角形判定を使う。

安定余裕 `margin` は次の符号付き距離である。

```text
margin =
  distance(c, boundary(B))   if c is inside B
 -distance(c, B)             if c is outside B
```

したがって、`margin > 0` なら COG 射影点は BOS 内部、`margin < 0` なら BOS 外部である。

## 6. COP の計算

COP は `src/cogsb/kinematics/cop.py` の `COPEstimator` が担当する。現在の実装は力板データを使う実測 COP ではなく、COG と加速度から物理投影で近似する。

質量を `m`、重力加速度を `g`、COG を `(x, y, z)`、COG 加速度を `(a_x, a_y, a_z)` とする。床反力の近似は次である。

```text
F_z = m * (g + a_z)
F_x = m * a_x
F_y = m * a_y
```

`F_z` は数値安定性のため最小 `1.0` に制限される。

COP 候補点は、水平力によるモーメント整合の 2D 近似として次で計算する。

```text
cop_x = x - z * F_x / F_z
cop_y = y + z * F_y / F_z
```

候補点が BOS 多角形内にあればそのまま COP とする。外にある場合は BOS 多角形上の最近点へ射影する。

摩擦制約も残差として評価する。摩擦係数を `mu` とすると、水平力がクーロン摩擦の範囲を超えた場合に残差を加算する。

```text
|F_x| <= mu * F_z
|F_y| <= mu * F_z
```

残差が大きいほど COP の信頼度は下がる。概念的には次である。

```text
confidence = 1 / (1 + residual)
```

ただし、多角形が作れない場合や残差がある場合は下限値を持つ。

## 7. 信頼度と欠損時の扱い

ランドマークが検出できないフレームでは `source_flags=["no_landmarks"]` として出力し、再構成・COG・BOS・COP の更新は行わない。

COG の信頼度は、再構成の変化量 `scale` と最適化残差 `residual` から次の形で計算される。

```text
confidence = 1 / (1 + scale + residual)
```

COG がゼロベクトルの場合は信頼度を `0` とする。

BOS 多角形が一時的に 3 点未満になった場合、前フレームの多角形があればそれを利用する。これは描画と判定の瞬間的な欠落を減らすためのフォールバックである。

## 8. 出力データ

出力は `src/cogsb/pipeline/writer.py` により JSON Lines として保存される。1 行が 1 フレームに対応し、主に次の情報を含む。

- `pose`: MediaPipe の姿勢推定結果
- `reconstructed`: 再構成後の関節列と平滑化後の関節列
- `cog`: COG、速度、加速度、信頼度
- `bos`: BOS 多角形、3D 多角形、接地信頼度、内外判定、安定余裕
- `cop`: COP、反力、信頼度、BOS 内判定、残差

## 9. 現在の前提と限界

本システムの COG / BOS / COP は、単眼画像または一般的な動画から得た姿勢推定結果に基づく推定値である。力板、床反力計、IMU、カメラキャリブレーション済み多視点計測などによる実測値ではない。

主な限界は次のとおりである。

- MediaPipe の world 座標系は実空間の絶対座標とは限らない。
- COP は実測ではなく、COG と加速度からの近似である。
- BOS は足部ランドマークに依存するため、足が隠れる、画面外に出る、靴や床面が誤検出される場合に不安定になる。
- 支持平面は足部点群の PCA / SVD で推定しており、厳密な床面推定ではない。
- 身体セグメントの質量比は一般的な人体モデルに基づく固定値で、個人差は標準では反映されない。

これらの制約により、本システムの値は臨床・安全判定用の確定値ではなく、姿勢変化やバランス傾向を視覚的に把握するための解析補助値として扱う。

## 参考資料

- `docs/bos_calculation_references.md`
- `docs/ground_plane_research_summary.md`
- `docs/3d_pose_temporal_research_summary.md`
- `src/cogsb/pipeline/engine.py`
- `src/cogsb/reconstruction/skeleton_optimizer.py`
- `src/cogsb/reconstruction/body_segments.py`
- `src/cogsb/kinematics/cog.py`
- `src/cogsb/kinematics/bos.py`
- `src/cogsb/kinematics/cop.py`
