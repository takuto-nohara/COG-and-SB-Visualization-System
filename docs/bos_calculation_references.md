# 支持基底面（BOS）計算の参考文献と適用方針

本修正で採用した支持基底面計算の根拠は、以下の査読付き論文に基づきます。

## 参照論文

- Hof, A. L., Gazendam, M. G. J., & Sinke, W. E. (2005). *The condition for dynamic stability*. Journal of Biomechanics, 38(1), 1-8. DOI: 10.1016/j.jbiomech.2004.03.025.
- Curtze, C., Buurke, T. J. W., & McCrum, C. (2024). *Notes on the margin of stability*. Journal of Biomechanics, 166, 112045. DOI: 10.1016/j.jbiomech.2024.112045.
- Millard, M., & Sloot, L. H. (2025). *A polygon model of the functional base-of-support during standing improves the accuracy of balance analysis*. Journal of Biomechanics, 192, 112927. DOI: 10.1016/j.jbiomech.2025.112927.
- Bretl, T., & Lall, S. (2008). *Testing Static Equilibrium for Legged Robots*. IEEE Transactions on Robotics, 24(4), 794–807. DOI: 10.1109/TRO.2008.2001360.

## 本実装への反映内容

- フッター点（足首・踵・第5中足骨）で支持平面を推定し、足部特徴点をこの平面へ射影してBOS多角形を構築。
- `compute_bos` が `cog_point` を受け取り、推定支持平面へ投影した `COG` で `inside/margin` 判定を行うように変更。
- `pipeline/engine.py` は生 `cog_vec` をそのまま使用するのではなく、`compute_bos` が返す `support_point_xy` ベースの `inside/margin` をそのまま利用。
- `pipeline`→`gui`→`visualization/draw` で `support_point_world` を受け渡し、`COG` と `BOS` の2D/3D描画を同一点群（支持平面上）へ揃える。

## 描画方針（本実装）

- BOS多角形は、2Dの接地輪郭だけでなく、可能なら `polygon_world`（3D空間上の対応頂点）を優先して描画基準に使う。
- `support_point_world` が存在する場合は、`COG` 表示を原点としてその点を用い、`cog.point` のXY平面投影とは独立して `support_polygon` と同一平面上で判定・可視化を行う。
- `COP` は現状 `compute_bos` の平面座標系を経由しているため、`COP` の3D復元は今後の改善課題とし、`BOS/COP` の同一平面整合性が優先される範囲で段階導入する。

## 本修正の意図

- 従来は「ランドマーク0番目（`nose`近傍）」を基準にしていたため、空間的に整合しにくい判定が起きていた。
- 先行研究の「支持基底面/BoS内部判定」を、足接地平面ベースで実装に近づけるために、COGと足接地点を同一平面へ投影する構成へ変更した。

## 追加参照

- Hof, A. L., Gazendam, M. G. J., & Sinke, W. E. (2005). *The condition for dynamic stability*. Journal of Biomechanics, 38(1), 1-8.
  - DOI: 10.1016/j.jbiomech.2004.03.025.
- Curtze, C., Buurke, T. J. W., & McCrum, C. (2024). *Notes on the margin of stability*. Journal of Biomechanics, 166, 112045.
  - DOI: 10.1016/j.jbiomech.2024.112045.
- Millard, M., & Sloot, L. H. (2025). *A polygon model of the functional base-of-support during standing improves the accuracy of balance analysis*. Journal of Biomechanics, 192, 112927.
  - DOI: 10.1016/j.jbiomech.2025.112927.
- Bretl, T., & Lall, S. (2008). *Testing Static Equilibrium for Legged Robots*. IEEE Transactions on Robotics, 24(4), 794–807.
  - DOI: 10.1109/TRO.2008.2001360.
