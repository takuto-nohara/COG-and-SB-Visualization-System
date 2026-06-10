# COG / SB Visualization System

本リポジトリは、映像ベースで人体の重心（COG）、支持基底面（BOS）、圧中心（COP）を推定し、
ライブ映像と録画動画のどちらでも解析できるよう設計したPython実装です。

## 特徴
- 入力: ライブカメラ(`--source live`) / 録画動画(`--source file --path`) / 静止画(`--source image --path`)対応
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

### GUI 起動
```bash
.\cogsb-gui.cmd
```
`cogsb-gui` 相当の起動スクリプトです。実行時に依存関係を解決 (`pip install -e .`) し、GUIを起動します（ネットワーク/権限のある環境で初回のみ時間がかかります）。

`.\cogsb-gui.cmd` 実行後、同じユーザー環境では通常 `cogsb-gui` が直接実行可能になります。PATH がすぐ反映されない場合は、次節の手順で追加してください。

`cogsb-gui` コマンドが認識されない場合は、まず `.\cogsb-gui.cmd` を 1 回実行してください。実行中に PATH 上の書き込み可能なディレクトリへ `cogsb-gui.cmd` を作成し、同一シェル内でも次回から `cogsb-gui` で起動できるようにします。

どうしても `cogsb-gui` が使えない場合は、警告に従って次のように PATH を追加します（永続化可）。
```powershell
$userScript = (python -c "import sysconfig; print(sysconfig.get_path('scripts', scheme='nt_user'))")
$env:PATH = "$userScript;$env:PATH"
[Environment]::SetEnvironmentVariable("Path", "$userScript;" + [Environment]::GetEnvironmentVariable("Path","User"), "User")
```

GUI では「写真」「動画」「リアルタイム映像」を選択して解析を開始できます。
画像/動画ファイルは画面内で選択し、3種類すべてで解析結果のフレーム上に重心（COG）と支持基底面（BOS）を同じ描画領域に重畳して表示します。

### 1コマンド起動
```bash
python -m pip install -e .
cogsb-start
```
`cogsb-start` はライブ映像を `--source live --source-id 0 --mode realtime` で起動します。

環境によっては `cogsb-start` が PATH に反映されない場合があります。その場合は以下でも起動できます。
```bash
python -m cogsb.cli.start
```

### mediapipe API 差分への対応
Python 3.13 では `mediapipe 0.10.35` 系で `mediapipe.solutions` が使えない場合があります。  
この環境向けに、`mediapipe.tasks` API を使って実行する際は `.task` モデルを初回起動時に自動取得します。

モデルファイルの場所を固定したい場合は以下を指定できます。
```bash
$env:COGSB_POSE_LANDMARKER_TASK_PATH = "C:\\path\\to\\pose_landmarker_lite.task"
python -m cogsb.cli.start
```

### 録画入力（mp4）
```bash
cogsb analyze --source file --path sample.mp4 --mode offline --output-dir outputs\run01
```

### 写真入力（jpg/png）
```bash
cogsb analyze --source image --path sample.jpg --mode offline --output-dir outputs\run_image
```

### 出力
- `outputs/frames.jsonl` に1フレームごとの解析結果を保存
- 各レコードには以下が含まれます。
  - `frame_idx`, `source_timestamp`, `source_type`
  - `cog`（重心、速度、加速度、信頼度）
  - `bos`（BOSポリゴン、左右接地確率、安定余裕）
  - `cop`（COP位置、反力、BOS内判定、残差）

### 可視化出力
```bash
cogsb visualize sample.mp4 outputs/frames.jsonl
cogsb visualize sample.mp4 outputs/frames.jsonl --output rendered.mp4
```

## 開発メモ
- `--mode realtime` は逐次処理を優先します。
- `--mode offline` は動画全体を最後まで処理するのに向きます。
- `--smoothness` を上げるとフレーム間の追従速度は下がります。

## 進捗
- フェーズ単位（live/file + MediaPipe、3D再構成、COG/BOS、COP、CLI、書き出し）で実装
