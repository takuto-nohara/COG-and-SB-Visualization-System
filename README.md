# COG / SB Visualization System

本リポジトリは、映像ベースで人体の重心（COG）、支持基底面（BOS）、圧中心（COP）を推定し、
ライブ映像と録画動画のどちらでも解析できるよう設計したPython実装です。

## 特徴
- 入力: ライブカメラ(`--source live`) / 録画動画(`--source file --path`)対応
- 推定: MediaPipe Pose
- 再構成: 体節長固定の骨格最適化（セグメント拘束 + 時系列平滑）
- 重心: Dempster/de Leva系係数によるセグメント合成
- BOS: 足部接地確率付きポリゴン
- COP: 反力・モーメント整合を使った物理拘束推定
- SMPL: オプションでSMPLフィッティング

## インストール
```bash
python -m pip install -e .
# SMPL/高機能最適化を使う場合
python -m pip install -e .[analysis]
```

## 実行例

### ライブ入力
```bash
cogsb analyze --source live --source-id 0 --mode realtime
```

### 録画入力（mp4）
```bash
cogsb analyze --source file --path sample.mp4 --mode offline --output-dir outputs\run01
```

### 出力
- `outputs/frames.jsonl` に1フレームごとの解析結果を保存
- 各レコードには以下が含まれます。
  - `frame_idx`, `source_timestamp`, `source_type`
  - `cog`（重心、速度、加速度、信頼度）
  - `bos`（BOSポリゴン、左右接地確率、安定余裕）
  - `cop`（COP位置、反力、BOS内判定、残差）

## 開発メモ
- `--mode realtime` は逐次処理を優先します。
- `--mode offline` は動画全体を最後まで処理するのに向きます。
- `--smoothness` を上げるとフレーム間の追従速度は下がります。

## 進捗
- フェーズ単位（live/file + MediaPipe、3D再構成、COG/BOS、COP、CLI、書き出し）で実装
