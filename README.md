# COG / SB Visualization System

COG/SB可視化システムは、`cogsb` CLI と GUI を中心に、人体姿勢推定から
COG・BOS・COPの解析結果を `outputs/frames.jsonl` へ保存し、必要に応じて動画へ重畳描画します。

## システム構成（現状）

- `cogsb.sources`: 入力の抽象化（ライブカメラ / 動画 / 画像）
- `cogsb.pose`: MediaPipe Pose推定
- `cogsb.reconstruction` / `cogsb.kinematics`: 骨格再構成と各種指標推定
- `cogsb.pipeline`: 解析エンジン
- `cogsb.visualization`: overlay描画と可視化
- `cogsb.cli`, `cogsb.gui`: CLI/GUI 入口

## インストール

```bash
python -m pip install -e .
# SMPL を有効化する場合（任意）
python -m pip install -e .[analysis]
```

## 実行

### 1) GUI起動（推奨）

```bash
.\cogsb-gui.cmd
```

`cogsb-gui.cmd` は同梱の起動ラッパーです。内部で環境整備後、`python -m cogsb.gui` を起動します。

起動後の手順:

- 入力種別を選択（写真 / 動画 / リアルタイム映像）
- 写真・動画はファイルを選択
- 「解析開始」を押して実行
- 解析結果は GUI 上のプレビューで COG / BOS / COP を重畳表示
- 出力は `outputs` 配下の新規フォルダ（`gui_session_YYYYMMDD_HHMMSS`）に `frames.jsonl` が保存されます。

### 2) 補助: CLI（必要時のみ）

- ライブ入力

```bash
cogsb analyze live --source-id 0 --mode realtime --output-dir outputs/live
```

- 動画入力

```bash
cogsb analyze file --path sample.mp4 --mode offline --output-dir outputs/video_run
```

- 画像入力

```bash
cogsb analyze image --path sample.jpg --mode offline --output-dir outputs/image_run
```

- 結果可視化（表示 or 出力保存）

```bash
cogsb visualize sample.mp4 outputs/video_run/frames.jsonl
cogsb visualize sample.mp4 outputs/video_run/frames.jsonl --output rendered.mp4
cogsb visualize sample.mp4 outputs/video_run/frames.jsonl --render-mode space3d --output rendered_3d.mp4
```

`render-mode`:

- `overlay`（既定）: 入力画像にCOG/BOS/COPなどを重ねて表示
- `space3d`: 3D空間上にセグメント・重心・COP/BOSを配置して表示

- ワンコマンドでライブ起動

```bash
cogsb-start
```

## 主要オプション

`cogsb analyze` の主な引数:

- `source`（位置引数）: `live | file | image`
- `--path`: `file/image` のとき必須
- `--source-id`: ライブ時のカメラID（既定: `0`）
- `--mode`: `realtime | offline`
- `--max-frames`: 処理フレーム数上限（任意）
- `--output-dir`: 出力先ディレクトリ（既定: `outputs`）
- `--smoothness`: 時系列平滑化係数（既定: `0.35`）
- `--smpl-model`: SMPLモデルパス（任意）
- `--enable-smpl`: SMPL推定を有効化（任意、`analysis` 依存必要）

## 出力

- 解析結果: `<output-dir>/frames.jsonl`
- 1レコードは `frame_idx`, `source_timestamp`, `source_type`, `cog`, `bos`, `cop` などを含みます。
- `cog`, `bos`, `cop` の主要項目:
  - `cog`: 重心、速度、加速度、信頼度
  - `bos`: ボリュームポリゴン、左右接地値、BOS内判定
  - `cop`: COP位置、反力/残差、BOS内判定

## ドキュメント

- [動作原理](docs/system_principles.md): COG / BOS / COP の計算式と処理フロー
- [BOS計算の参考文献](docs/bos_calculation_references.md): 支持基底面計算の根拠と適用方針

## 既知補足

- MediaPipe は環境により `solutions` / `tasks` API が使い分けられます。
  Tasks API を使う場合は `COGSB_POSE_LANDMARKER_TASK_PATH` で `.task`モデルを固定可能です。
