# 3D姿勢推定（2D→3D）と時系列処理の調査サマリ

## 目的
本ノートは、本リポジトリの3D描画モードでの品質低下要因を、  
- 「2Dランドマークから3D関節を推定する研究」  
- 「フレーム単位推定が引き起こすジッタと、時系列拘束の有効性」  
の観点で調査し、実装へ反映した内容を残すためのものです。

## 参考文献（査読付き）

1. Julieta Martinez, Rayat Hossain, Javier Romero, James J. Little,  
   **A Simple Yet Effective Baseline for 3D Human Pose Estimation**（ICCV 2017）  
   DOI: 10.1109/ICCV.2017.288  
   - 2Dランドマークを入力として、比較的簡素なモデルで高精度な3D復元が可能であることを示す。  
   - 2D画像→3Dの分解戦略（2D推定＋3D持ち上げ）が有効であることを示す基礎的な議論として参照。  
   - Link: https://openaccess.thecvf.com/content_ICCV_2017/papers/Martinez_A_Simple_yet_ICCV_2017_paper.pdf

2. Dario Pavllo et al.,  
   **3D Human Pose Estimation in Video With Temporal Convolutions and Semi-Supervised Training**（CVPR 2019）  
   - 2Dキーポイント系列に対する時系列畳み込み（Temporal Convolution）で3D姿勢を推定し、単純なRNNより高精度化。  
   - 時系列全体での整合性を利用することでフレーム単位推定より性能向上。  
   - Link: https://openaccess.thecvf.com/content_CVPR_2019/papers/Pavllo_3D_Human_Pose_Estimation_in_Video_With_Temporal_Convolutions_and_CVPR_2019_paper.pdf

3. Mir Rayat Imtiaz Hossain, James J. Little,  
   **Exploiting temporal information for 3D pose estimation**（ECCV 2018, 予稿はarXiv:1711.08585）  
   - 「フレームごとの独立推定は、各フレームの誤差が独立であるため時間的一貫性が崩れ、ジッタを生む」ことを明示。  
   - LSTMベースの時系列エンコーダ/デコーダで時間拘束を入れ、12%以上の性能改善を報告。  
   - Link: https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Mir_Rayat_Imtiaz_Hossain_Exploiting_temporal_information_ECCV_2018_paper.pdf

4. Weipeng Xu et al.,  
   **MonoPerfCap: Human Performance Capture from Monocular Video**（ACM TOG / SIGGRAPH 2018）  
   - 単眼動画における時間的一貫性のある3D人体運動推定のため、バッチ単位の推定と軌道制約を採用。  
   - 単眼由来の深度曖昧性を時系列拘束で解消する重要性を示唆。  
   - Link: https://arxiv.org/abs/1708.02136

5. 本リポジトリ既存の地面平面整合化の調査ノート  
   - ground plane推定に関する参考とする。  
   - Link: docs/ground_plane_research_summary.md

## 実装反映内容

1. 3D描画が推定結果を参照しない問題を解消  
   - `src/cogsb/pipeline/engine.py`で算出された`reconstructed`の`joints_smooth`を  
     GUI描画オーバーレイへ渡すように変更（`pose3d_joints`）。
   - `src/cogsb/gui.py::_to_pose_payload`で`output.reconstructed`を有効活用。
   - `src/cogsb/visualization/draw.py::_collect_world_points_for_3d`で、`pose3d_joints`を
     `world_landmarks`より優先して使用。
   - 結果として3D空間表示は、MediaPipe世界座標の未補正値ではなく、再構成後平滑化済み3D推定を描画。

2. フレーム単位推定由来ジッタ低減のための時系列拘束  
   - `src/cogsb/reconstruction/skeleton_optimizer.py`で、再構成最適化に「前フレーム速度拘束」を追加。  
     - `prev_velocity`（前フレーム関節速度）を残差項として追加。
     - `dt`（フレーム間時間）を考慮して`p_t ≈ p_{t-1} + v_{t-1}*dt`を促進。
   - `src/cogsb/pipeline/engine.py`でフレーム間の推定速度を計算し、次フレームの再構成へ入力。
   - 初期フレーム、検出欠損フレームでは速度状態をリセット。

## 今後の拡張案

本リポジトリの制約（画像/映像サイズ、推論量）を前提に、次の段階としては以下を検討します。

- 可能なら最近接の短いフレーム列（例: 2〜3秒窓）を使って滑らか化（加速度拘束・角速度制約）を加える。
- 遮蔽/検出欠損時は速度拘束を減衰し、ランドマーク信頼度に応じて動的に重み切替を行う。
- `src/cogsb/pipeline/writer.py`にも再構成データの保存を追加し、再現性（ログ/デバッグ）を上げる。
