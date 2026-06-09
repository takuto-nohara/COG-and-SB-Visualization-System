# COG / SB Visualization System

このリポジトリは、映像入力（ライブ/録画）から人体の骨格を推定し、
- 重心（COG）
- 支持基底面（BOS）
- 圧中心点（COP）
を同時に推定し、可視化結果を出力する実験基盤です。

## 主要構成
- 入力: ライブカメラ / 動画ファイル
- 推定: MediaPipe Pose
- 再構成: 骨格長固定最適化＋SMPL補正
- 力学: Dempster/de Leva系体節係数ベースのCOG、足部接地確率BOS、物理拘束ベースCOP

## 実行
```bash
python -m pip install -e .
# 追加アルゴリズムを使う場合
python -m pip install -e .[analysis]
```

```bash
cogsb analyze --source live --source-id 0 --mode realtime
cogsb analyze --source file --path sample.mp4 --mode offline
```

最初の実装では処理パイプラインを通して安定した入力可否と推定ログを確保し、
段階的に最適化精度を上げます。
