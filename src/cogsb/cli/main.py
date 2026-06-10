from __future__ import annotations

from typing import Any, Optional
from types import SimpleNamespace

try:
    import typer  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - import guard for incomplete environments
    class _TyperFallback:
        class BadParameter(RuntimeError):
            pass

        @staticmethod
        def _default_arg(*args: Any, **kwargs: Any) -> Any:
            if args:
                return args[0]
            return kwargs.get("default")

        @staticmethod
        def Argument(*args: Any, **kwargs: Any) -> Any:
            return _TyperFallback._default_arg(*args, **kwargs)

        @staticmethod
        def Option(*args: Any, **kwargs: Any) -> Any:
            return _TyperFallback._default_arg(*args, **kwargs)

        class Typer:
            def command(self, *_: Any, **__: Any):
                def decorator(func: Any) -> Any:
                    return func

                return decorator

            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("typer is not installed")

    typer = SimpleNamespace(Argument=_TyperFallback.Argument, Option=_TyperFallback.Option, BadParameter=_TyperFallback.BadParameter, Typer=_TyperFallback.Typer)  # type: ignore[assignment]

from cogsb.core import PipelineConfig, PipelineMode, SourceType
from cogsb.pipeline.engine import AnalysisEngine
from cogsb.pose import MediaPipePoseEstimator
from cogsb.smpl.smpl_fitter import SMPLFitter
from cogsb.sources import LiveCameraSource, RecordedImageSource, RecordedVideoSource
from cogsb.visualization.render import render_video
from cogsb.visualization.draw import RENDER_MODE_OPTIONS


def analyze(
    source: str = typer.Argument("file", help="live / file / image"),
    path: Optional[str] = typer.Option(None, help="入力メディアのパス（source=file/image時必須）"),
    source_id: int = typer.Option(0, help="ライブ入力のデバイスID"),
    mode: str = typer.Option("realtime", help="realtime または offline"),
    max_frames: Optional[int] = typer.Option(None, help="最大処理フレーム数"),
    output_dir: str = typer.Option("outputs", help="出力先ディレクトリ"),
    smoothness: float = typer.Option(0.35, help="時間平滑化率"),
    smpl_model: Optional[str] = typer.Option(None, help="SMPLモデルディレクトリ（任意）"),
    enable_smpl: bool = typer.Option(False, help="SMPLフィッティングを有効化"),
) -> None:
    if max_frames is not None and not isinstance(max_frames, int):
        max_frames = None

    if source not in {"live", "file", "image"}:
        raise typer.BadParameter("source は live / file / image を指定してください")
    if source in {"file", "image"} and not path:
        raise typer.BadParameter("file/image入力時は --path が必要です")
    assert path is not None

    mode_enum = PipelineMode.REALTIME if mode == "realtime" else PipelineMode.OFFLINE
    if source == "live":
        source_type = SourceType.LIVE
    elif source == "file":
        source_type = SourceType.FILE
    else:
        source_type = SourceType.IMAGE

    cfg = PipelineConfig(
        mode=mode_enum,
        source_type=source_type,
        output_dir=output_dir,
        max_frames=max_frames,
        smpl_enabled=enable_smpl,
        smpl_model_path=smpl_model,
    )

    if source == "live":
        source_obj = LiveCameraSource(camera_index=source_id, output_rgb=False)
    elif source == "file":
        source_obj = RecordedVideoSource(video_path=path, output_rgb=False)
    else:
        source_obj = RecordedImageSource(image_path=path, output_rgb=False)

    estimator = MediaPipePoseEstimator(output_rgb=False, static_image_mode=source == "image")
    smpl_fitter = SMPLFitter(
        enabled=enable_smpl,
        model_path=smpl_model,
        gender="neutral",
        use_gpu=False,
    )

    engine = AnalysisEngine(cfg, smpl_fitter=smpl_fitter)
    engine.smoother.alpha = smoothness
    engine.run(source_obj, estimator, max_frames=max_frames)


def visualize(
    video_path: str = typer.Argument(..., help="可視化対象の動画パス"),
    jsonl: str = typer.Argument(..., help="Analyze時に生成された outputs/frames.jsonl"),
    output: Optional[str] = typer.Option(None, help="出力mp4のパス（未指定なら即時表示）"),
    render_mode: str = typer.Option("overlay", help="描画モード: overlay（画像重畳）または space3d（3D表示）"),
) -> None:
    mode = str(render_mode).lower()
    if mode not in RENDER_MODE_OPTIONS:
        raise typer.BadParameter("render-mode は overlay または space3d を指定してください")

    render_video(video_path=video_path, jsonl_path=jsonl, output_path=output, render_mode=mode)


def main() -> None:
    app = typer.Typer()
    app.command()(analyze)
    app.command()(visualize)
    app()


if __name__ == "__main__":
    main()
