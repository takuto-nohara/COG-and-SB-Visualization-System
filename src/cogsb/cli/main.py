from __future__ import annotations

from typing import Optional

import typer

from cogsb.core import PipelineConfig, PipelineMode, SourceType
from cogsb.pipeline.engine import AnalysisEngine
from cogsb.pose import MediaPipePoseEstimator
from cogsb.smpl.smpl_fitter import SMPLFitter
from cogsb.sources import LiveCameraSource, RecordedVideoSource


def analyze(
    source: str = typer.Argument("file", help="live または file"),
    path: Optional[str] = typer.Option(None, help="録画ファイルのパス（source=file時必須）"),
    source_id: int = typer.Option(0, help="ライブ入力のデバイスID"),
    mode: str = typer.Option("realtime", help="realtime または offline"),
    max_frames: Optional[int] = typer.Option(None, help="最大処理フレーム数"),
    output_dir: str = typer.Option("outputs", help="出力先ディレクトリ"),
    smoothness: float = typer.Option(0.35, help="時間平滑化率"),
    smpl_model: Optional[str] = typer.Option(None, help="SMPLモデルディレクトリ（任意）"),
    enable_smpl: bool = typer.Option(False, help="SMPLフィッティングを有効化"),
) -> None:
    if source not in {"live", "file"}:
        raise typer.BadParameter("source は live または file を指定してください")
    if source == "file" and not path:
        raise typer.BadParameter("file入力時は --path が必要です")

    mode_enum = PipelineMode.REALTIME if mode == "realtime" else PipelineMode.OFFLINE
    cfg = PipelineConfig(
        mode=mode_enum,
        source_type=SourceType.LIVE if source == "live" else SourceType.FILE,
        output_dir=output_dir,
        max_frames=max_frames,
        smpl_enabled=enable_smpl,
        smpl_model_path=smpl_model,
    )

    if source == "live":
        source_obj = LiveCameraSource(camera_index=source_id, output_rgb=False)
    else:
        source_obj = RecordedVideoSource(video_path=path, output_rgb=False)

    estimator = MediaPipePoseEstimator(output_rgb=False)
    smpl_fitter = SMPLFitter(
        enabled=enable_smpl,
        model_path=smpl_model,
        gender="neutral",
        use_gpu=False,
    )

    engine = AnalysisEngine(cfg, smpl_fitter=smpl_fitter)
    engine.smoother.alpha = smoothness
    engine.run(source_obj, estimator, max_frames=max_frames)


def main() -> None:
    app = typer.Typer()
    app.command()(analyze)
    app()


if __name__ == "__main__":
    main()
