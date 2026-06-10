from __future__ import annotations

import base64
import queue
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from cogsb.core import PipelineConfig, PipelineMode, SourceType
from cogsb.pipeline.engine import AnalysisEngine
from cogsb.pose import MediaPipePoseEstimator
from cogsb.smpl.smpl_fitter import SMPLFitter
from cogsb.sources import LiveCameraSource, RecordedImageSource, RecordedVideoSource
from cogsb.visualization.draw import RENDER_MODE_OPTIONS, RENDER_MODE_OVERLAY, RENDER_MODE_SPACE3D, draw_frame_overlay


class COGSBGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("COG / SB 可視化GUI")
        self.root.geometry("1320x820")
        self.root.minsize(960, 640)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.source_var = tk.StringVar(value="image")
        self.source_path_var = tk.StringVar()
        self.render_mode_var = tk.StringVar(value=RENDER_MODE_OVERLAY)
        self.render_mode_radios: list[ttk.Radiobutton] = []
        self.camera_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="起動準備完了")
        self.info_var = tk.StringVar(value="")
        self._photo_image: Optional[tk.PhotoImage] = None
        self._preview_photo_image: Optional[tk.PhotoImage] = None
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._queue: queue.Queue[Tuple[str, Any]] = queue.Queue(maxsize=12)
        self._last_output_path = Path("outputs")
        self._latest_render_frame: Optional[Any] = None
        self._latest_render_output = None
        self._space3d_view = {
            "yaw": -40.0,
            "pitch": 22.0,
            "zoom": 1.0,
            "pan_x": 0.0,
            "pan_y": 0.0,
        }
        self._space3d_drag = {
            "mode": None,
            "start_x": 0,
            "start_y": 0,
            "base_yaw": -40.0,
            "base_pitch": 22.0,
            "base_pan_x": 0.0,
            "base_pan_y": 0.0,
        }

        self._build_ui()
        self._queue_poll()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        split = ttk.PanedWindow(container, orient=tk.HORIZONTAL)
        split.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(split, width=320, padding=10)
        sidebar.pack_propagate(False)
        split.add(sidebar, weight=0)

        preview_area = ttk.Frame(split, padding=(10, 10, 0, 10))
        split.add(preview_area, weight=1)

        controls = ttk.LabelFrame(sidebar, text="解析設定")
        controls.pack(fill=tk.X, pady=(0, 10))

        source_row = ttk.Frame(controls)
        source_row.pack(fill=tk.X, padx=8, pady=(6, 2))
        ttk.Label(source_row, text="入力種類").pack(side=tk.LEFT)

        for value, caption in (
            ("image", "写真"),
            ("file", "動画"),
            ("live", "リアルタイム映像"),
        ):
            ttk.Radiobutton(
                source_row,
                text=caption,
                value=value,
                variable=self.source_var,
                command=self._toggle_controls,
            ).pack(side=tk.LEFT, padx=(8, 4))

        path_row = ttk.Frame(controls)
        path_row.pack(fill=tk.X, padx=8, pady=(2, 2))
        ttk.Label(path_row, text="ファイル").pack(side=tk.LEFT)
        self.path_entry = ttk.Entry(path_row, textvariable=self.source_path_var)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.browse_btn = ttk.Button(path_row, text="参照", command=self._pick_file)
        self.browse_btn.pack(side=tk.LEFT, padx=(0, 8))

        camera_row = ttk.Frame(controls)
        camera_row.pack(fill=tk.X, padx=8, pady=(2, 2))
        ttk.Label(camera_row, text="カメラID").pack(side=tk.LEFT)
        self.camera_entry = ttk.Entry(camera_row, textvariable=self.camera_var, width=6)
        self.camera_entry.pack(side=tk.LEFT, padx=8)
        ttk.Label(camera_row, text="(live時のみ)").pack(side=tk.LEFT)

        button_row = ttk.Frame(controls)
        button_row.pack(fill=tk.X, padx=8, pady=(4, 6))
        self.start_btn = ttk.Button(button_row, text="解析開始", command=self._start_analysis)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.stop_btn = ttk.Button(button_row, text="停止", command=self._stop_analysis, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)

        render_mode_row = ttk.Frame(controls)
        render_mode_row.pack(fill=tk.X, padx=8, pady=(4, 6))
        ttk.Label(render_mode_row, text="描画モード").pack(side=tk.LEFT)
        for value, caption in (
            (RENDER_MODE_OVERLAY, "画像重ね"),
            (RENDER_MODE_SPACE3D, "3D空間"),
        ):
            radio = ttk.Radiobutton(
                render_mode_row,
                text=caption,
                value=value,
                variable=self.render_mode_var,
                command=self._on_render_mode_change,
            )
            radio.pack(side=tk.LEFT, padx=(8, 4))
            self.render_mode_radios.append(radio)

        reset_row = ttk.Frame(controls)
        reset_row.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Button(reset_row, text="3D視点リセット", command=self._reset_space3d_view).pack(side=tk.LEFT)

        status_row = ttk.Frame(controls)
        status_row.pack(fill=tk.X, padx=8, pady=(4, 2))
        ttk.Label(status_row, text="状態:").pack(side=tk.LEFT)
        ttk.Label(status_row, textvariable=self.status_var, wraplength=240).pack(side=tk.LEFT, padx=8)

        source_preview = ttk.LabelFrame(sidebar, text="選択ファイルプレビュー")
        source_preview.pack(fill=tk.X, pady=(0, 10))
        self.source_path_label = ttk.Label(source_preview, text="未選択", wraplength=280, justify=tk.LEFT)
        self.source_path_label.pack(fill=tk.X, padx=8, pady=(8, 4))
        self.source_preview_canvas = tk.Canvas(
            source_preview, bg="#1a1a1a", width=280, height=160, highlightthickness=0
        )
        self.source_preview_canvas.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._set_sidebar_message("入力ファイルを選択すると、左側に1コマ目を表示します。")

        log_frame = ttk.LabelFrame(sidebar, text="実行ログ")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_listbox = tk.Listbox(log_frame, height=9)
        self.log_listbox.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._log("起動しました")

        metadata_panel = ttk.LabelFrame(sidebar, text="解析メトリクス")
        metadata_panel.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(metadata_panel, textvariable=self.info_var, wraplength=280, justify=tk.LEFT).pack(
            fill=tk.X,
            padx=8,
            pady=8,
        )

        preview = ttk.LabelFrame(
            preview_area, text="解析結果プレビュー（2D重ね合わせ / 3D空間表示）"
        )
        preview.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(preview, bg="#111111", width=1024, height=576, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_rotate_start)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_drag_end)
        self.canvas.bind("<ButtonPress-3>", self._on_canvas_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_canvas_drag_end)
        self.canvas.bind("<MouseWheel>", self._on_canvas_wheel)
        self.canvas.bind("<Button-4>", self._on_canvas_wheel)
        self.canvas.bind("<Button-5>", self._on_canvas_wheel)

        self._toggle_controls()

    def _toggle_controls(self) -> None:
        source = self.source_var.get()
        if source == "live":
            self.path_entry.config(state=tk.DISABLED)
            self.browse_btn.config(state=tk.DISABLED)
            self.camera_entry.config(state=tk.NORMAL)
        else:
            self.path_entry.config(state=tk.NORMAL)
            self.browse_btn.config(state=tk.NORMAL)
            self.camera_entry.config(state=tk.DISABLED)

    def _pick_file(self) -> None:
        source = self.source_var.get()
        if source == "image":
            filetypes = [
                ("Image", "*.jpg"),
                ("Image", "*.jpeg"),
                ("Image", "*.png"),
                ("Image", "*.bmp"),
                ("Image", "*.webp"),
                ("All", "*.*"),
            ]
            path = filedialog.askopenfilename(filetypes=filetypes)
        else:
            filetypes = [
                ("Video", "*.mp4"),
                ("Video", "*.mov"),
                ("Video", "*.avi"),
                ("Video", "*.mkv"),
                ("Video", "*.webm"),
                ("All", "*.*"),
            ]
            path = filedialog.askopenfilename(filetypes=filetypes)

        if path:
            self.source_path_var.set(path)
            self.source_path_label.config(text=path)
            self._preview_selected_file(path, source)
        else:
            self.source_path_label.config(text="未選択")
            self._set_sidebar_message("入力ファイルを選択すると、左側に1コマ目を表示します。")

    def _log(self, message: str) -> None:
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_listbox.insert(tk.END, f"[{timestamp}] {message}")
            if self.log_listbox.size() > 120:
                self.log_listbox.delete(0, self.log_listbox.size() - 120)
            self.log_listbox.yview_moveto(1.0)
        except Exception:
            pass

    def _set_sidebar_message(self, message: str) -> None:
        self.source_preview_canvas.delete("all")
        self.source_preview_canvas.create_text(
            140,
            80,
            text=message,
            fill="#a0a0a0",
            width=250,
            justify=tk.CENTER,
            anchor=tk.CENTER,
            font=("TkDefaultFont", 10),
        )

    @staticmethod
    def _extract_frame_array(frame_any: Any):
        if hasattr(frame_any, "frame"):
            frame_any = frame_any.frame
        if frame_any is None:
            return None
        try:
            return frame_any.copy()
        except Exception:
            if hasattr(frame_any, "shape"):
                return frame_any
        return None

    def _draw_on_canvas(
        self,
        canvas: tk.Canvas,
        frame: Any,
        max_w: int,
        max_h: int,
        is_main: bool = False,
    ) -> None:
        if frame is None:
            raise ValueError("frame is None")
        if not hasattr(frame, "shape"):
            raise ValueError("Unsupported frame object")

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim != 3:
            raise ValueError(f"Unsupported frame dimension: {frame.ndim}")

        if frame.shape[2] != 3:
            raise ValueError(f"Unsupported channel count: {frame.shape[2]}")

        frame_h, frame_w = frame.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError(f"Invalid frame size: {frame_w}x{frame_h}")

        frame_to_show = frame
        if frame_to_show.dtype != np.uint8:
            frame_to_show = np.clip(frame_to_show, 0, 255).astype(np.uint8)

        view_w = max(1, int(max_w))
        view_h = max(1, int(max_h))
        max_w = max(1, view_w)
        max_h = max(1, view_h)

        h, w = frame_to_show.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            frame_to_show = cv2.resize(
                frame_to_show,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        encode_ok, png_data = cv2.imencode(".png", frame_to_show)
        if not encode_ok:
            raise RuntimeError("PNGエンコードに失敗しました。")
        png_b64 = base64.b64encode(png_data).decode("ascii")
        image = tk.PhotoImage(data=png_b64)

        if is_main:
            self._photo_image = image
        else:
            self._preview_photo_image = image

        canvas.delete("all")
        target_w, target_h = frame_to_show.shape[1], frame_to_show.shape[0]
        x = max(0, (view_w - target_w) // 2)
        y = max(0, (view_h - target_h) // 2)
        canvas.create_image(x, y, image=image, anchor=tk.NW)

    def _preview_selected_file(self, path: str, source_type: str) -> None:
        self._set_sidebar_message("プレビュー読み込み中…")
        frame: Optional[Any] = None

        try:
            if source_type == "image":
                frame = cv2.imread(path)
            else:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError("動画を開けませんでした。")
                ok, first_frame = cap.read()
                cap.release()
                if not ok:
                    raise RuntimeError("動画の1コマ目を取得できませんでした。")
                frame = first_frame

            if frame is None:
                raise RuntimeError("プレビューに使えるフレームがありません。")

            self.source_preview_canvas.update_idletasks()
            view_w = self.source_preview_canvas.winfo_width()
            view_h = self.source_preview_canvas.winfo_height()
            if view_w <= 1 or view_h <= 1:
                self.root.after(25, lambda: self._draw_sidebar_preview(frame))
                return

            self._draw_on_canvas(self.source_preview_canvas, frame, max_w=view_w - 2, max_h=view_h - 2, is_main=False)
        except Exception as exc:
            self._set_sidebar_message(f"サイドバープレビュー失敗: {exc}")
            self._log(f"サイドバープレビュー失敗: {exc}")

    def _draw_sidebar_preview(self, frame: Any) -> None:
        try:
            self.source_preview_canvas.update_idletasks()
            view_w = self.source_preview_canvas.winfo_width()
            view_h = self.source_preview_canvas.winfo_height()
            if view_w <= 1 or view_h <= 1:
                self.root.after(25, lambda: self._draw_sidebar_preview(frame))
                return
            self._draw_on_canvas(
                self.source_preview_canvas,
                frame,
                max_w=view_w - 2,
                max_h=view_h - 2,
                is_main=False,
            )
        except Exception as exc:
            self._set_sidebar_message(f"サイドバープレビュー失敗: {exc}")

    @staticmethod
    def _to_pose_payload(output, frame_shape: Tuple[int, int]) -> dict[str, Any]:
        if output.pose is None:
            return {}

        landmarks = []
        for lm in output.pose.landmarks:
            landmarks.append(
                {
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(lm.visibility),
                }
            )

        payload: dict[str, Any] = {
            "pose": {
                "landmarks": landmarks,
                "shape": [frame_shape[1], frame_shape[0]],
            }
        }
        if output.pose.world_landmarks:
            payload["pose"]["world_landmarks"] = [
                {
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(lm.visibility),
                }
                for lm in output.pose.world_landmarks
            ]

        return payload

    @staticmethod
    def _to_float_pair(value: Optional[tuple]) -> Optional[Tuple[float, float]]:
        if value is None or len(value) < 2:
            return None
        return float(value[0]), float(value[1])

    def _to_overlay(self, output, frame_shape: Tuple[int, int]) -> dict[str, Any]:
        payload = self._to_pose_payload(output, frame_shape)

        if output.cog is not None:
            cog_xy = self._to_float_pair(output.cog.cog[:2])
            if cog_xy is not None:
                payload["cog"] = {
                    "point": [float(cog_xy[0]), float(cog_xy[1])],
                    "confidence": float(output.cog.confidence),
                }

        if output.bos is not None:
            payload["bos"] = {
                "polygon": [[float(x), float(y)] for x, y in output.bos.polygon],
                "inside": output.bos.inside_cog,
                "inside_cog": output.bos.inside_cog,
                "support_area": float(output.bos.support_area),
            }

        if output.cop is not None and output.cop.cop is not None:
            cp = self._to_float_pair(output.cop.cop)
            if cp is not None:
                payload["cop"] = {
                    "cop": [float(cp[0]), float(cp[1])],
                    "within_bos": bool(output.cop.within_bos),
                }

        return payload

    def _render_frame(self, frame, output) -> None:
        try:
            frame = self._extract_frame_array(frame)
            if frame is None:
                raise ValueError("描画フレームを取得できませんでした。")

            source_frame = frame.copy()
            frame_h, frame_w = frame.shape[:2]
            overlay = self._to_overlay(output, (frame_h, frame_w))
            render_mode = self.render_mode_var.get()
            if render_mode not in RENDER_MODE_OPTIONS:
                render_mode = RENDER_MODE_OVERLAY
            render_state = self._space3d_view if render_mode == RENDER_MODE_SPACE3D else None
            frame = draw_frame_overlay(
                source_frame,
                overlay,
                render_mode=render_mode,
                render_state=render_state,
            )

            self._latest_render_frame = source_frame
            self._latest_render_output = output

            self.canvas.update_idletasks()
            view_w = self.canvas.winfo_width()
            view_h = self.canvas.winfo_height()
            if view_w <= 1 or view_h <= 1:
                self.root.after(25, lambda: self._render_frame(frame.copy(), output))
                return

            lines = [
                f"frame: {output.frame_idx if output is not None else '-'}",
                f"source: {output.source_type.value if output and output.source_type is not None else '-'}",
            ]

            if output.cog is not None:
                cog_x, cog_y, cog_z = output.cog.cog
                lines.append(f"COG: ({cog_x:.3f}, {cog_y:.3f}, {cog_z:.3f})")
                if output.cog.confidence is not None:
                    lines.append(f"COG conf: {output.cog.confidence:.2f}")

            if output.bos is not None:
                if output.bos.support_area is not None:
                    lines.append(f"BOS area: {output.bos.support_area:.4f}")
                if output.bos.inside_cog is not None:
                    lines.append(f"COG in BOS: {output.bos.inside_cog}")

            if output.cop is not None and output.cop.cop is not None:
                lines.append(f"COP: ({output.cop.cop[0]:.3f}, {output.cop.cop[1]:.3f})")

            self.info_var.set(" | ".join(lines))
            self._draw_on_canvas(self.canvas, frame, max_w=view_w - 4, max_h=view_h - 4, is_main=True)
        except Exception as exc:
            self._log(f"描画エラー: {exc}")
            self.status_var.set("描画エラー")
            self.root.after(0, lambda: messagebox.showerror("描画エラー", str(exc)))

    def _on_render_mode_change(self) -> None:
        self._redraw_latest_frame()

    def _reset_space3d_view(self) -> None:
        self._space3d_view = {
            "yaw": -40.0,
            "pitch": 22.0,
            "zoom": 1.0,
            "pan_x": 0.0,
            "pan_y": 0.0,
        }
        self._redraw_latest_frame()

    def _redraw_latest_frame(self) -> None:
        if self._latest_render_frame is None or self._latest_render_output is None:
            return
        self._render_frame(self._latest_render_frame, self._latest_render_output)

    def _on_canvas_rotate_start(self, event: tk.Event[tk.Misc]) -> None:
        if self.render_mode_var.get() != RENDER_MODE_SPACE3D:
            return
        self._space3d_drag["mode"] = "rotate"
        self._space3d_drag["start_x"] = event.x
        self._space3d_drag["start_y"] = event.y
        self._space3d_drag["base_yaw"] = self._space3d_view["yaw"]
        self._space3d_drag["base_pitch"] = self._space3d_view["pitch"]

    def _on_canvas_pan_start(self, event: tk.Event[tk.Misc]) -> None:
        if self.render_mode_var.get() != RENDER_MODE_SPACE3D:
            return
        self._space3d_drag["mode"] = "pan"
        self._space3d_drag["start_x"] = event.x
        self._space3d_drag["start_y"] = event.y
        self._space3d_drag["base_pan_x"] = self._space3d_view["pan_x"]
        self._space3d_drag["base_pan_y"] = self._space3d_view["pan_y"]

    def _on_canvas_drag(self, event: tk.Event[tk.Misc]) -> None:
        if self.render_mode_var.get() != RENDER_MODE_SPACE3D:
            return
        mode = self._space3d_drag.get("mode")
        if mode is None:
            return

        dx = float(event.x - self._space3d_drag["start_x"])
        dy = float(event.y - self._space3d_drag["start_y"])

        if mode == "rotate":
            yaw = self._space3d_drag["base_yaw"] + dx * 0.35
            pitch = self._space3d_drag["base_pitch"] - dy * 0.35
            pitch = max(-85.0, min(85.0, pitch))
            self._space3d_view["yaw"] = yaw
            self._space3d_view["pitch"] = pitch
        elif mode == "pan":
            self._space3d_view["pan_x"] = self._space3d_drag["base_pan_x"] + dx
            self._space3d_view["pan_y"] = self._space3d_drag["base_pan_y"] + dy

        self._redraw_latest_frame()

    def _on_canvas_drag_end(self, event: tk.Event[tk.Misc]) -> None:
        self._space3d_drag["mode"] = None

    def _on_canvas_wheel(self, event: tk.Event[tk.Misc]) -> None:
        if self.render_mode_var.get() != RENDER_MODE_SPACE3D:
            return

        delta = 0
        if hasattr(event, "num") and event.num in (4, 5):
            delta = 1 if event.num == 4 else -1
        else:
            if hasattr(event, "delta") and event.delta != 0:
                delta = 1 if event.delta > 0 else -1

        if delta == 0:
            return

        factor = 1.12 if delta > 0 else 0.89
        zoom = self._space3d_view["zoom"] * factor
        self._space3d_view["zoom"] = max(0.08, min(6.0, zoom))
        self._redraw_latest_frame()

    def _set_controls(self, running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        self.start_btn.config(state=state)
        self.browse_btn.config(state=tk.DISABLED if running or self.source_var.get() == "live" else tk.NORMAL)
        self.path_entry.config(state=tk.DISABLED if self.source_var.get() == "live" or running else tk.NORMAL)
        radio_state = tk.DISABLED if running else tk.NORMAL
        for radio in self.render_mode_radios:
            radio.config(state=radio_state)
        if running:
            self.stop_btn.config(state=tk.NORMAL)
            self.camera_entry.config(state=tk.NORMAL if self.source_var.get() == "live" else tk.DISABLED)
        else:
            self.stop_btn.config(state=tk.DISABLED)
            self._toggle_controls()

    def _start_analysis(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return

        source_type = self.source_var.get()
        source_path = self.source_path_var.get().strip()
        if source_type in {"image", "file"} and not source_path:
            messagebox.showerror("入力不足", "写真/動画のときはファイルを選択してください。")
            return

        if source_type == "live":
            try:
                source_id = int(self.camera_var.get() or "0")
            except ValueError:
                messagebox.showerror("入力エラー", "カメラIDは数値で入力してください。")
                return
        else:
            source_id = 0

        self._stop_event = threading.Event()
        self._set_controls(True)
        self.status_var.set("解析を開始しています")
        self.info_var.set("")
        self._log("解析開始")

        mode = PipelineMode.REALTIME if source_type == "live" else PipelineMode.OFFLINE
        max_frames = None if source_type != "image" else 1
        out_dir = Path("outputs") / f"gui_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cfg = PipelineConfig(
            mode=mode,
            source_type=SourceType(source_type),
            output_dir=str(out_dir),
            max_frames=max_frames,
            smpl_enabled=False,
        )
        self._last_output_path = out_dir

        def runner() -> None:
            estimator: Optional[MediaPipePoseEstimator] = None
            try:
                if source_type == "live":
                    source = LiveCameraSource(camera_index=source_id, output_rgb=False)
                elif source_type == "file":
                    source = RecordedVideoSource(video_path=source_path, output_rgb=False)
                else:
                    source = RecordedImageSource(image_path=source_path, output_rgb=False)

                estimator = MediaPipePoseEstimator(
                    output_rgb=False,
                    static_image_mode=(source_type == "image"),
                )
                smpl_fitter = SMPLFitter(
                    enabled=False,
                    model_path=None,
                    gender="neutral",
                    use_gpu=False,
                )
                engine = AnalysisEngine(cfg, smpl_fitter=smpl_fitter)
                engine.smoother.alpha = 0.35

                self._queue.put(("status", f"出力先: {out_dir}"))
                self._log(f"出力先: {out_dir}")

                def on_frame(frame, frame_output):
                    if self._stop_event.is_set():
                        return False
                    frame_array = self._extract_frame_array(frame)
                    if frame_array is None:
                        self._log("解析結果フレームの取得に失敗しました")
                        return True
                    try:
                        self._queue.put(("frame", frame_array.copy(), frame_output), block=False)
                    except queue.Full:
                        try:
                            self._queue.get_nowait()
                        except Exception:
                            pass
                        try:
                            self._queue.put(("frame", frame_array.copy(), frame_output), block=False)
                        except Exception:
                            return False
                    return True

                engine.run(
                    source,
                    estimator,
                    max_frames=max_frames,
                    on_frame=on_frame,
                    stop_event=self._stop_event,
                )

                if self._stop_event.is_set():
                    self._queue.put(("status", "解析を停止しました"))
                    self._log("解析を停止しました")
                else:
                    self._queue.put(("status", f"解析完了: {out_dir}"))
                    self._log(f"解析完了: {out_dir}")
            except Exception as exc:
                self._log(f"解析エラー: {exc}")
                self._queue.put(("error", str(exc)))
            finally:
                self._queue.put(("finished",))
                if estimator is not None:
                    estimator.close()

        self._worker = threading.Thread(target=runner, daemon=True)
        self._worker.start()

    def _stop_analysis(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            return
        self._stop_event.set()
        self.status_var.set("停止中")

    def _on_close(self) -> None:
        self._stop_event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=0.5)
        self.root.destroy()

    def _queue_poll(self) -> None:
        try:
            while True:
                item = self._queue.get_nowait()
                kind = item[0]
                if kind == "frame":
                    _, frame, output = item
                    self._render_frame(frame, output)
                elif kind == "status":
                    self.status_var.set(str(item[1]))
                elif kind == "error":
                    self.status_var.set(f"エラー: {item[1]}")
                    messagebox.showerror("解析エラー", str(item[1]))
                elif kind == "finished":
                    self._set_controls(False)
                    self._worker = None
        except queue.Empty:
            pass

        self.root.after(30, self._queue_poll)


def main() -> None:
    COGSBGUI().root.mainloop()


__all__ = ["main"]


if __name__ == "__main__":
    main()
