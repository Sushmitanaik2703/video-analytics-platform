"""
Smart CCTV Monitoring System - Dynamic Video Analytics Platform (Desktop UI)

Features:
1. Video Upload: User can upload MP4, AVI, or MOV files from computer, or select from sample videos.
2. Dynamic Polygon Drawing: Draw, reset, and save one or multiple custom restricted zones directly on the video's first frame.
3. Real-Time Video Playback on Screen: Video actively moves and animates on the canvas during detection with all YOLO bounding boxes, labels, track IDs, and red alert banners.
4. Real-Time Incident Logging: Live KPI cards (Total, Person, Vehicle, Active Alerts) and Incident Table update immediately on every frame.
5. Live Security Alert Banner: Lights up red immediately when an intrusion is detected.
6. Full YOLOv8s + ByteTrack Pipeline: Detects and tracks persons and vehicles with immediate red alerts and automatic exit clearing.
7. Terminal Logging: Real-time playback timestamps and structured alert output.
8. Comprehensive Results Dashboard: KPI cards, incident audit table, screenshot gallery, and one-click annotated video playback.
"""

from __future__ import annotations

import csv
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intrusion_detection import IntrusionDetector, RestrictedZone, process_video_stream
from integrated_pipeline import IntegratedPipeline, process_integrated_stream
from incident_database import MongoIncidentService, serialize_incident


class SmartCCTVApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Smart CCTV Monitoring System — Dynamic Video Analytics Platform")
        self.root.geometry("1420x920")
        self.root.minsize(1120, 740)

        # Configure Dark / Surveillance Theme
        self.bg_color = "#181a1f"
        self.panel_bg = "#21252b"
        self.card_bg = "#282c34"
        self.accent_color = "#4c8bf5"
        self.accent_hover = "#357ae8"
        self.text_color = "#abb2bf"
        self.text_bright = "#ffffff"
        self.alert_color = "#e06c75"
        self.success_color = "#98c379"
        self.warning_color = "#e5c07b"

        self.root.configure(bg=self.bg_color)
        self._setup_styles()

        # Video & Canvas State
        self.video_path: Path | None = None
        self.first_frame_bgr: np.ndarray | None = None
        self.photo_img: ImageTk.PhotoImage | None = None
        self.live_photo: ImageTk.PhotoImage | None = None
        self.canvas_img_id: int | None = None

        self.orig_w: int = 0
        self.orig_h: int = 0
        self.fps: float = 30.0
        self.total_frames: int = 0
        self.disp_w: int = 0
        self.disp_h: int = 0
        self.offset_x: int = 0
        self.offset_y: int = 0

        # Polygon Zone Drawing State
        self.saved_zones: list[dict] = []  # List of dicts with id, name, points, classes
        self.current_poly_pts: list[tuple[int, int]] = []  # Canvas pixel coords
        self.temp_mouse_pos: tuple[int, int] | None = None

        # Execution State
        self.is_processing = False
        self.stop_event = threading.Event()
        self.ui_queue: queue.Queue = queue.Queue(maxsize=2)
        self.last_results: dict | None = None
        self.annotated_video_path: Path | None = None
        self.saved_screenshots: list[str] = []
        self.last_ui_draw_time = 0.0
        self._incident_table_signature: tuple | None = None
        self._database_results: queue.Queue = queue.Queue()
        self._incident_store = MongoIncidentService(self._database_results)
        self._saved_incidents: dict[str, dict] = {}
        self._current_incident_history: list[dict] = []
        self._persistence_signatures: dict[str, str] = {}

        # Build UI Components
        self._build_header()
        self._build_body()
        self._build_statusbar()

        self.root.bind("<Destroy>", self._on_root_destroy, add="+")
        self._incident_store.load_incidents()
        self.root.after(100, self._poll_database_results)

        # Check for pre-loaded samples
        self._populate_sample_videos()

    def _on_root_destroy(self, event):
        if event.widget is self.root:
            self._incident_store.close()

    def _poll_database_results(self):
        try:
            while True:
                message_type, payload = self._database_results.get_nowait()
                if message_type == "loaded":
                    for incident in payload:
                        incident_key = incident.get("incident_key")
                        if incident_key:
                            self._saved_incidents.setdefault(incident_key, incident)
                    self._refresh_incident_dashboard()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_database_results)

    def _set_current_incidents(self, incident_history: list[dict]):
        video_filename = self.video_path.name if self.video_path else None
        current_incidents = []
        for incident in incident_history:
            current = dict(incident)
            document = serialize_incident(current, video_filename)
            if document is not None:
                incident_key = document["incident_key"]
                current["incident_key"] = incident_key
                self._saved_incidents[incident_key] = current
                signature = json.dumps(document, sort_keys=True, default=str)
                if self._persistence_signatures.get(incident_key) != signature:
                    self._incident_store.save_incident(document)
                    self._persistence_signatures[incident_key] = signature
            current_incidents.append(current)
        self._current_incident_history = current_incidents

    def _all_incidents(self) -> list[dict]:
        incidents = dict(self._saved_incidents)
        for incident in self._current_incident_history:
            incident_key = incident.get("incident_key")
            if incident_key:
                incidents[incident_key] = incident
        return list(incidents.values())

    def _refresh_incident_dashboard(
        self,
        *,
        active_alert_count: int | None = None,
        fps: float | None = None,
    ):
        incidents = self._all_incidents()
        vehicle_classes = {"car", "motorcycle", "bus", "truck"}
        total_incidents = len(incidents)
        person_incidents = sum(
            1
            for incident in incidents
            if incident.get("class_name") == "person"
            and incident.get("type") != "UNATTENDED_BAG"
        )
        vehicle_incidents = sum(
            1
            for incident in incidents
            if incident.get("class_name") in vehicle_classes
        )
        bag_incidents = sum(
            1 for incident in incidents if incident.get("type") == "UNATTENDED_BAG"
        )

        self.kpi_total.val_lbl.config(text=str(total_incidents))
        self.kpi_person.val_lbl.config(text=str(person_incidents))
        self.kpi_vehicle.val_lbl.config(text=str(vehicle_incidents))
        self.kpi_bag.val_lbl.config(text=str(bag_incidents))
        if active_alert_count is not None:
            self.kpi_active.val_lbl.config(text=str(active_alert_count))
        if fps is not None:
            self.kpi_fps.val_lbl.config(text=f"{fps:.1f} FPS")

        incident_signature = tuple(
            (
                incident.get("incident_key"),
                incident.get("incident_id"),
                incident.get("type"),
                incident.get("track_id"),
                incident.get("class_name"),
                incident.get("zone_name"),
                incident.get("entry_time"),
                incident.get("exit_time"),
                incident.get("duration_seconds"),
                incident.get("status"),
            )
            for incident in incidents
        )
        if incident_signature != self._incident_table_signature:
            self._sync_incident_table(self._current_incident_history, incidents)
            self._incident_table_signature = incident_signature
        return total_incidents, person_incidents, vehicle_incidents, bag_incidents

    def _setup_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=self.panel_bg, foreground=self.text_color, font=("Segoe UI", 9))
        style.configure("TFrame", background=self.panel_bg)
        style.configure("TLabelframe", background=self.panel_bg, foreground=self.text_bright, font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", background=self.panel_bg, foreground=self.text_bright)

        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), background=self.accent_color, foreground=self.text_bright)
        style.map("Accent.TButton", background=[("active", self.accent_hover)])

        style.configure("Danger.TButton", font=("Segoe UI", 9, "bold"), background="#d9534f", foreground=self.text_bright)
        style.map("Danger.TButton", background=[("active", "#c9302c")])

        style.configure("Success.TButton", font=("Segoe UI", 10, "bold"), background=self.success_color, foreground="#1e2227")
        style.map("Success.TButton", background=[("active", "#87b168")])

        style.configure("Treeview", background=self.card_bg, foreground=self.text_bright, fieldbackground=self.card_bg, rowheight=24)
        style.map("Treeview", background=[("selected", self.accent_color)])
        style.configure("Treeview.Heading", background=self.panel_bg, foreground=self.text_bright, font=("Segoe UI", 9, "bold"))

        style.configure("Horizontal.TProgressbar", background=self.accent_color, troughcolor=self.card_bg)

    def _build_header(self):
        header_frame = tk.Frame(self.root, bg=self.card_bg, height=65, relief="flat")
        header_frame.pack(side="top", fill="x")

        title_box = tk.Frame(header_frame, bg=self.card_bg)
        title_box.pack(side="left", padx=20, pady=10)

        title_lbl = tk.Label(
            title_box,
            text="SMART CCTV MONITORING SYSTEM",
            font=("Segoe UI", 14, "bold"),
            fg=self.text_bright,
            bg=self.card_bg,
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            title_box,
            text="Dynamic Video Upload • Interactive Polygon Zone Selection • Real-Time Intrusion Analytics",
            font=("Segoe UI", 9),
            fg=self.accent_color,
            bg=self.card_bg,
        )
        sub_lbl.pack(anchor="w")

        badge_box = tk.Frame(header_frame, bg=self.card_bg)
        badge_box.pack(side="right", padx=20, pady=12)

        self.status_badge = tk.Label(
            badge_box,
            text="● SYSTEM READY",
            font=("Segoe UI", 9, "bold"),
            fg=self.success_color,
            bg="#1d2d22",
            padx=12,
            pady=4,
            relief="groove",
        )
        self.status_badge.pack(side="right")

    def _build_body(self):
        body_paned = tk.PanedWindow(self.root, orient="horizontal", bg=self.bg_color, bd=0, sashwidth=5)
        body_paned.pack(fill="both", expand=True, padx=10, pady=10)

        # ------------------- LEFT SIDEBAR: CONTROLS & CONFIG -------------------
        left_panel = tk.Frame(body_paned, bg=self.panel_bg, width=370)
        body_paned.add(left_panel, minsize=330)

        sidebar_canvas = tk.Canvas(left_panel, bg=self.panel_bg, highlightthickness=0)
        sidebar_scroll = ttk.Scrollbar(left_panel, orient="vertical", command=sidebar_canvas.yview)
        scrollable_frame = tk.Frame(sidebar_canvas, bg=self.panel_bg)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")),
        )
        sidebar_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=350)
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)

        sidebar_canvas.pack(side="left", fill="both", expand=True)
        sidebar_scroll.pack(side="right", fill="y")

        # 1. Video Upload Section
        sec_upload = ttk.LabelFrame(scrollable_frame, text=" 1. Video Source & Upload ", padding=10)
        sec_upload.pack(fill="x", padx=10, pady=6)

        btn_browse = ttk.Button(sec_upload, text="📁  Upload Video (MP4, AVI, MOV)", style="Accent.TButton", command=self.on_upload_video)
        btn_browse.pack(fill="x", pady=4)

        lbl_or = tk.Label(sec_upload, text="— OR SELECT SAMPLE VIDEO —", font=("Segoe UI", 8), fg="#7f848e", bg=self.panel_bg)
        lbl_or.pack(pady=2)

        self.sample_combo = ttk.Combobox(sec_upload, state="readonly", values=["(Select Sample...)"])
        self.sample_combo.pack(fill="x", pady=2)
        self.sample_combo.bind("<<ComboboxSelected>>", self.on_sample_selected)

        self.lbl_video_info = tk.Label(
            sec_upload,
            text="No video loaded. Click Upload Video to begin.",
            font=("Consolas", 8),
            fg=self.warning_color,
            bg=self.card_bg,
            justify="left",
            padx=8,
            pady=6,
            relief="sunken",
            wraplength=310,
        )
        self.lbl_video_info.pack(fill="x", pady=6)

        # 2. Active AI Monitoring Features Section
        sec_features = ttk.LabelFrame(scrollable_frame, text=" 2. Active AI Monitoring Features ", padding=10)
        sec_features.pack(fill="x", padx=10, pady=6)

        self.chk_feat_intrusion = tk.BooleanVar(value=True)
        self.chk_feat_vehicle = tk.BooleanVar(value=True)
        self.chk_feat_bag = tk.BooleanVar(value=True)
        self.chk_feat_fight = tk.BooleanVar(value=False)
        self.fight_threshold_var = tk.DoubleVar(value=0.55)

        self.chk_feat_intrusion.trace_add("write", lambda *args: self._on_feature_toggle())
        self.chk_feat_vehicle.trace_add("write", lambda *args: self._on_feature_toggle())
        self.chk_feat_bag.trace_add("write", lambda *args: self._on_feature_toggle())
        self.chk_feat_fight.trace_add("write", lambda *args: self._on_feature_toggle())

        ttk.Checkbutton(sec_features, text="🚨 Restricted-Zone Intrusion Detection", variable=self.chk_feat_intrusion).pack(anchor="w", pady=2)
        ttk.Checkbutton(sec_features, text="🚗 Vehicle Detection & Tracking", variable=self.chk_feat_vehicle).pack(anchor="w", pady=2)
        ttk.Checkbutton(sec_features, text="💼 Unattended-Bag Detection (Automatic)", variable=self.chk_feat_bag).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            sec_features,
            text="🥊 Physical Altercation Detection",
            variable=self.chk_feat_fight,
        ).pack(anchor="w", pady=2)

        fight_threshold_row = tk.Frame(sec_features, bg=self.panel_bg)
        fight_threshold_row.pack(fill="x", pady=(5, 0))
        tk.Label(
            fight_threshold_row,
            text="Possible-altercation threshold:",
            font=("Segoe UI", 8),
            fg=self.text_color,
            bg=self.panel_bg,
        ).pack(side="left")
        self.lbl_fight_threshold_value = tk.Label(
            fight_threshold_row,
            text="55%",
            font=("Segoe UI", 8, "bold"),
            fg=self.text_bright,
            bg=self.panel_bg,
        )
        self.lbl_fight_threshold_value.pack(side="right")
        ttk.Scale(
            sec_features,
            from_=0.35,
            to=0.95,
            variable=self.fight_threshold_var,
            command=self._on_fight_threshold_change,
        ).pack(fill="x", pady=(1, 0))

        lbl_feat_note = tk.Label(
            sec_features,
            text=(
                "💡 Bag and vehicle tracking run full-frame. Fight alerts show a "
                "window-level possibility for human review; no individuals are identified."
            ),
            font=("Segoe UI", 7, "italic"),
            fg=self.text_color,
            bg=self.panel_bg,
            wraplength=310,
            justify="left",
        )
        lbl_feat_note.pack(fill="x", pady=(4, 0))

        # 3. Dynamic Restricted-Zone Selection
        sec_zones = ttk.LabelFrame(scrollable_frame, text=" 3. Dynamic Restricted-Zone Selection ", padding=10)
        sec_zones.pack(fill="x", padx=10, pady=6)

        inst_text = (
            "📌 Instructions (Required for Zone Intrusion only):\n"
            "• Click on video frame to place polygon points.\n"
            "• Right-click to undo last point.\n"
            "• Click 'Save Zone' or double-click to close."
        )
        lbl_inst = tk.Label(sec_zones, text=inst_text, font=("Segoe UI", 8), fg=self.text_color, bg=self.panel_bg, justify="left")
        lbl_inst.pack(anchor="w", pady=2)

        zf_row1 = tk.Frame(sec_zones, bg=self.panel_bg)
        zf_row1.pack(fill="x", pady=4)
        tk.Label(zf_row1, text="Zone Name:", font=("Segoe UI", 8, "bold"), fg=self.text_bright, bg=self.panel_bg).pack(side="left")
        self.zone_name_entry = ttk.Entry(zf_row1, width=20)
        self.zone_name_entry.insert(0, "Restricted Zone 1")
        self.zone_name_entry.pack(side="right", fill="x", expand=True, padx=4)

        cls_box = tk.Frame(sec_zones, bg=self.panel_bg)
        cls_box.pack(fill="x", pady=4)
        tk.Label(cls_box, text="Target Classes:", font=("Segoe UI", 8, "bold"), fg=self.text_bright, bg=self.panel_bg).pack(anchor="w")

        cls_inner = tk.Frame(cls_box, bg=self.panel_bg)
        cls_inner.pack(fill="x", pady=2)
        self.chk_person_var = tk.BooleanVar(value=True)
        self.chk_vehicle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cls_inner, text="Person", variable=self.chk_person_var).pack(side="left", padx=4)
        ttk.Checkbutton(cls_inner, text="Vehicles", variable=self.chk_vehicle_var).pack(side="left", padx=4)

        btn_box = tk.Frame(sec_zones, bg=self.panel_bg)
        btn_box.pack(fill="x", pady=6)
        ttk.Button(btn_box, text="💾 Save Zone", command=self.on_save_zone).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(btn_box, text="↺ Reset Current", command=self.on_reset_current_points).pack(side="left", fill="x", expand=True, padx=2)

        tk.Label(sec_zones, text="Configured Zones:", font=("Segoe UI", 8, "bold"), fg=self.text_bright, bg=self.panel_bg).pack(anchor="w", pady=(6, 2))
        self.zones_listbox = tk.Listbox(sec_zones, height=4, bg=self.card_bg, fg=self.text_bright, font=("Consolas", 8), selectmode="single")
        self.zones_listbox.pack(fill="x", pady=2)

        zone_actions = tk.Frame(sec_zones, bg=self.panel_bg)
        zone_actions.pack(fill="x", pady=2)
        ttk.Button(zone_actions, text="🗑 Delete Selected", command=self.on_delete_selected_zone).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(zone_actions, text="Clear All", command=self.on_clear_all_zones).pack(side="left", fill="x", expand=True, padx=2)

        # 4. Detection Parameters Section
        sec_params = ttk.LabelFrame(scrollable_frame, text=" 4. Detection & Model Settings ", padding=10)
        sec_params.pack(fill="x", padx=10, pady=6)

        m_row = tk.Frame(sec_params, bg=self.panel_bg)
        m_row.pack(fill="x", pady=2)
        tk.Label(m_row, text="Model:", font=("Segoe UI", 8), fg=self.text_color, bg=self.panel_bg).pack(side="left")
        self.model_combo = ttk.Combobox(m_row, state="readonly", values=["yolov8s.pt", "yolov8n.pt"], width=14)
        self.model_combo.set("yolov8s.pt")
        self.model_combo.pack(side="right")

        conf_row = tk.Frame(sec_params, bg=self.panel_bg)
        conf_row.pack(fill="x", pady=4)
        self.lbl_conf = tk.Label(conf_row, text="Confidence: 0.45", font=("Segoe UI", 8), fg=self.text_color, bg=self.panel_bg)
        self.lbl_conf.pack(side="left")
        self.scale_conf = ttk.Scale(conf_row, from_=0.10, to=0.90, value=0.45, command=self._on_conf_change)
        self.scale_conf.pack(side="right", fill="x", expand=True, padx=8)

        cf_row = tk.Frame(sec_params, bg=self.panel_bg)
        cf_row.pack(fill="x", pady=2)
        tk.Label(cf_row, text="Confirm Frames:", font=("Segoe UI", 8), fg=self.text_color, bg=self.panel_bg).pack(side="left")
        self.spin_confirm = ttk.Spinbox(cf_row, from_=1, to=10, width=5)
        self.spin_confirm.set(2)
        self.spin_confirm.pack(side="right")

        sched_row = tk.Frame(sec_params, bg=self.panel_bg)
        sched_row.pack(fill="x", pady=2)
        tk.Label(sched_row, text="Schedule:", font=("Segoe UI", 8), fg=self.text_color, bg=self.panel_bg).pack(side="left")
        self.sched_combo = ttk.Combobox(sched_row, state="readonly", values=["auto", "daytime", "night"], width=10)
        self.sched_combo.set("auto")
        self.sched_combo.pack(side="right")

        # Unattended Bag Parameters
        bag_param_frame = tk.LabelFrame(sec_params, text="Unattended Bag Parameters", font=("Segoe UI", 8, "bold"), fg=self.warning_color, bg=self.panel_bg, padx=6, pady=4)
        bag_param_frame.pack(fill="x", pady=(6, 2))

        b_row1 = tk.Frame(bag_param_frame, bg=self.panel_bg)
        b_row1.pack(fill="x", pady=2)
        tk.Label(b_row1, text="Stationary Dwell (s):", font=("Segoe UI", 8), fg=self.text_color, bg=self.panel_bg).pack(side="left")
        self.spin_stationary = ttk.Spinbox(b_row1, from_=1.0, to=60.0, increment=1.0, width=5)
        self.spin_stationary.set(15.0)
        self.spin_stationary.pack(side="right")

        b_row2 = tk.Frame(bag_param_frame, bg=self.panel_bg)
        b_row2.pack(fill="x", pady=2)
        tk.Label(b_row2, text="Owner Distance (px):", font=("Segoe UI", 8), fg=self.text_color, bg=self.panel_bg).pack(side="left")
        self.spin_person_dist = ttk.Spinbox(b_row2, from_=50, to=500, increment=25, width=5)
        self.spin_person_dist.set(150)
        self.spin_person_dist.pack(side="right")

        # 5. Start & Stop Detection Section
        sec_run = ttk.LabelFrame(scrollable_frame, text=" 5. Execute Pipeline ", padding=10)
        sec_run.pack(fill="x", padx=10, pady=8)

        self.btn_start = ttk.Button(sec_run, text="▶  START DETECTION", style="Success.TButton", command=self.on_start_detection)
        self.btn_start.pack(fill="x", pady=4)

        self.btn_stop = ttk.Button(sec_run, text="⏹ Stop Detection", style="Danger.TButton", command=self.on_stop_detection, state="disabled")
        self.btn_stop.pack(fill="x", pady=2)

        self.progress_bar = ttk.Progressbar(sec_run, orient="horizontal", mode="determinate", style="Horizontal.TProgressbar")
        self.progress_bar.pack(fill="x", pady=6)

        self.lbl_progress = tk.Label(sec_run, text="Status: Ready", font=("Segoe UI", 8), fg=self.text_color, bg=self.panel_bg)
        self.lbl_progress.pack(anchor="w")

        # ------------------- RIGHT MAIN AREA: REAL-TIME MONITORING & DETAILS -------------------
        right_panel = tk.Frame(body_paned, bg=self.bg_color)
        body_paned.add(right_panel, minsize=700)

        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill="both", expand=True)

        # TAB 1: Live CCTV Monitor & Real-Time Incidents (UNIFIED VIEW)
        self.tab_monitor = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_monitor, text="  Live CCTV Monitor & Real-Time Analytics  ")

        self._build_live_monitor_tab(self.tab_monitor)

        # TAB 2: Full Audit Log & Screenshot Gallery
        self.tab_audit = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_audit, text="  Full Audit Log & Screenshots Gallery  ")

        self._build_audit_gallery_tab(self.tab_audit)

    def _build_live_monitor_tab(self, parent: ttk.Frame):
        tab_container = tk.Frame(parent, bg=self.bg_color)
        tab_container.pack(fill="both", expand=True, padx=6, pady=6)

        # 1. Top Security Alert Banner
        self.live_alert_banner = tk.Label(
            tab_container,
            text="● ALL RESTRICTED ZONES SECURE — READY FOR MONITORING",
            font=("Segoe UI", 10, "bold"),
            bg="#1d2d22",
            fg=self.success_color,
            pady=6,
            relief="groove",
        )
        self.live_alert_banner.pack(fill="x", pady=(0, 6))

        # 2. Main Video Screen (Canvas)
        canvas_outer = tk.Frame(tab_container, bg="#0d1117", relief="solid", bd=1)
        canvas_outer.pack(fill="both", expand=True, pady=(0, 4))

        self.canvas = tk.Canvas(canvas_outer, bg="#0d1117", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<Double-Button-1>", lambda e: self.on_save_zone())
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Configure>", lambda e: self._redraw_canvas())

        # Dedicated Notification Panel below video
        self.notif_panel = tk.Frame(tab_container, bg="#1a1d24", relief="groove", bd=1)
        self.notif_panel.pack(fill="x", pady=(0, 6))

        self.lbl_notif_title = tk.Label(
            self.notif_panel,
            text="🔔 SECURITY NOTIFICATIONS:",
            font=("Segoe UI", 8, "bold"),
            fg="#61afef",
            bg="#1a1d24",
        )
        self.lbl_notif_title.pack(side="left", padx=(8, 4), pady=4)

        self.lbl_notif_text = tk.Label(
            self.notif_panel,
            text="System ready. Draw restricted zone or select features to start monitoring.",
            font=("Segoe UI", 8),
            fg=self.text_bright,
            bg="#1a1d24",
            anchor="w",
        )
        self.lbl_notif_text.pack(side="left", fill="x", expand=True, padx=4, pady=4)

        # 3. Real-Time KPI Cards Row
        kpi_row = tk.Frame(tab_container, bg=self.bg_color)
        kpi_row.pack(fill="x", pady=(0, 6))

        self.kpi_total = self._create_kpi_card(kpi_row, "TOTAL INCIDENTS", "0", self.alert_color)
        self.kpi_total.pack(side="left", fill="x", expand=True, padx=2)

        self.kpi_person = self._create_kpi_card(kpi_row, "PERSON INTRUSIONS", "0", self.warning_color)
        self.kpi_person.pack(side="left", fill="x", expand=True, padx=2)

        self.kpi_vehicle = self._create_kpi_card(kpi_row, "VEHICLE INTRUSIONS", "0", self.accent_color)
        self.kpi_vehicle.pack(side="left", fill="x", expand=True, padx=2)

        self.kpi_bag = self._create_kpi_card(kpi_row, "UNATTENDED BAGS", "0", "#e06c75")
        self.kpi_bag.pack(side="left", fill="x", expand=True, padx=2)

        self.kpi_active = self._create_kpi_card(kpi_row, "ACTIVE ALERTS", "0", "#ff6b6b")
        self.kpi_active.pack(side="left", fill="x", expand=True, padx=2)

        self.kpi_fps = self._create_kpi_card(kpi_row, "PROCESSING SPEED", "-- FPS", self.success_color)
        self.kpi_fps.pack(side="left", fill="x", expand=True, padx=2)

        # 4. Real-Time Incident Table (Live Updates During Video Playback)
        table_frame = ttk.LabelFrame(tab_container, text=" Live Intrusion Incidents (Real-Time Updates) ", padding=4)
        table_frame.pack(fill="both", expand=False, pady=(0, 4))
        table_frame.configure(height=180)

        columns = ("id", "type", "track_id", "class", "zone", "entry", "exit", "duration", "status")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=5, selectmode="browse")

        self.tree.heading("id", text="#")
        self.tree.heading("type", text="Incident Type")
        self.tree.heading("track_id", text="Person / Track")
        self.tree.heading("class", text="Class")
        self.tree.heading("zone", text="Zone")
        self.tree.heading("entry", text="Entry Time")
        self.tree.heading("exit", text="Exit Time")
        self.tree.heading("duration", text="Duration")
        self.tree.heading("status", text="Status")

        self.tree.column("id", width=35, anchor="center")
        self.tree.column("type", width=130, anchor="w")
        self.tree.column("track_id", width=65, anchor="center")
        self.tree.column("class", width=75, anchor="center")
        self.tree.column("zone", width=150, anchor="w")
        self.tree.column("entry", width=110, anchor="center")
        self.tree.column("exit", width=110, anchor="center")
        self.tree.column("duration", width=75, anchor="center")
        self.tree.column("status", width=85, anchor="center")

        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        # 5. Bottom Controls & Action Buttons
        action_bar = tk.Frame(tab_container, bg=self.bg_color)
        action_bar.pack(fill="x", pady=(2, 0))

        self.btn_play_video = ttk.Button(
            action_bar,
            text="🎬 Play Annotated Output Video",
            style="Accent.TButton",
            command=self.on_play_annotated_video,
            state="disabled",
        )
        self.btn_play_video.pack(side="left", padx=4)

        self.btn_open_folder = ttk.Button(
            action_bar,
            text="📂 Open Output Folder",
            command=self.on_open_output_folder,
            state="disabled",
        )
        self.btn_open_folder.pack(side="left", padx=4)

        self.btn_export_log = ttk.Button(
            action_bar,
            text="📑 Export Incident Report (CSV)",
            command=self.on_export_incident_report,
            state="disabled",
        )
        self.btn_export_log.pack(side="left", padx=4)

        self.lbl_shots_info = tk.Label(
            action_bar,
            text="Annotated outputs saved in runs/user_uploads/",
            font=("Segoe UI", 8),
            fg=self.text_color,
            bg=self.bg_color,
        )
        self.lbl_shots_info.pack(side="right", padx=6)

    def _build_audit_gallery_tab(self, parent: ttk.Frame):
        container = tk.Frame(parent, bg=self.bg_color)
        container.pack(fill="both", expand=True, padx=8, pady=8)

        # Top: Audit Table
        audit_frame = ttk.LabelFrame(container, text=" Complete Permanent Intrusion Audit Log ", padding=6)
        audit_frame.pack(fill="both", expand=True, pady=(0, 8))

        columns = ("id", "type", "track_id", "class", "zone", "entry", "exit", "duration", "status")
        self.tree_audit = ttk.Treeview(audit_frame, columns=columns, show="headings", selectmode="browse")

        self.tree_audit.heading("id", text="#")
        self.tree_audit.heading("type", text="Incident Type")
        self.tree_audit.heading("track_id", text="Track ID")
        self.tree_audit.heading("class", text="Class")
        self.tree_audit.heading("zone", text="Zone")
        self.tree_audit.heading("entry", text="Entry Time")
        self.tree_audit.heading("exit", text="Exit Time")
        self.tree_audit.heading("duration", text="Duration")
        self.tree_audit.heading("status", text="Status")

        self.tree_audit.column("id", width=40, anchor="center")
        self.tree_audit.column("type", width=140, anchor="w")
        self.tree_audit.column("track_id", width=70, anchor="center")
        self.tree_audit.column("class", width=80, anchor="center")
        self.tree_audit.column("zone", width=160, anchor="w")
        self.tree_audit.column("entry", width=120, anchor="center")
        self.tree_audit.column("exit", width=120, anchor="center")
        self.tree_audit.column("duration", width=80, anchor="center")
        self.tree_audit.column("status", width=90, anchor="center")

        audit_scroll = ttk.Scrollbar(audit_frame, orient="vertical", command=self.tree_audit.yview)
        self.tree_audit.configure(yscrollcommand=audit_scroll.set)
        self.tree_audit.pack(side="left", fill="both", expand=True)
        audit_scroll.pack(side="right", fill="y")

        # Bottom: Screenshots Viewer
        gallery_frame = ttk.LabelFrame(container, text=" Event Screenshots (Click to View) ", padding=6)
        gallery_frame.pack(fill="x", expand=False)
        gallery_frame.configure(height=140)

        self.gallery_canvas = tk.Canvas(gallery_frame, bg=self.card_bg, height=110, highlightthickness=0)
        self.gallery_scroll = ttk.Scrollbar(gallery_frame, orient="horizontal", command=self.gallery_canvas.xview)
        self.gallery_inner = tk.Frame(self.gallery_canvas, bg=self.card_bg)

        self.gallery_inner.bind("<Configure>", lambda e: self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all")))
        self.gallery_canvas.create_window((0, 0), window=self.gallery_inner, anchor="nw")
        self.gallery_canvas.configure(xscrollcommand=self.gallery_scroll.set)

        self.gallery_canvas.pack(fill="x", expand=True)
        self.gallery_scroll.pack(fill="x")
        self.thumb_refs: list[ImageTk.PhotoImage] = []

    def _create_kpi_card(self, parent: tk.Widget, label: str, value: str, val_color: str) -> tk.Frame:
        card = tk.Frame(parent, bg=self.card_bg, padx=10, pady=8, relief="groove")
        tk.Label(card, text=label, font=("Segoe UI", 7, "bold"), fg=self.text_color, bg=self.card_bg).pack(anchor="w")
        val_lbl = tk.Label(card, text=value, font=("Segoe UI", 15, "bold"), fg=val_color, bg=self.card_bg)
        val_lbl.pack(anchor="w")
        card.val_lbl = val_lbl
        return card

    def _build_statusbar(self):
        status_bar = tk.Frame(self.root, bg=self.panel_bg, height=26)
        status_bar.pack(side="bottom", fill="x")

        self.status_text = tk.Label(status_bar, text="Ready. Select or upload a video file to begin.", font=("Segoe UI", 8), fg=self.text_color, bg=self.panel_bg)
        self.status_text.pack(side="left", padx=12)

        self.lbl_active_zones_count = tk.Label(status_bar, text="Active Zones: 0", font=("Segoe UI", 8, "bold"), fg=self.warning_color, bg=self.panel_bg)
        self.lbl_active_zones_count.pack(side="right", padx=14)

    def _on_conf_change(self, val):
        self.lbl_conf.config(text=f"Confidence: {float(val):.2f}")

    def _populate_sample_videos(self):
        samples = [
            ("ABODA CCTV (aboda_video2.avi)", PROJECT_ROOT / "datasets" / "samples" / "real_world_tests" / "aboda_video2.avi"),
            ("Industrial Factory Worker Breach (industrial_factory_worker_breach.mp4)", PROJECT_ROOT / "datasets" / "industrial_security_demo" / "industrial_factory_worker_breach.mp4"),
            ("Logistics Bay Vehicle & Workers (industrial_logistics_bay_vehicle_workers.mp4)", PROJECT_ROOT / "datasets" / "industrial_security_demo" / "industrial_logistics_bay_vehicle_workers.mp4"),
            ("CCTV Restricted Lane (cctv_restricted_lane_8101862.mp4)", PROJECT_ROOT / "datasets" / "vehicle_intrusion" / "cctv_restricted_lane_8101862.mp4"),
            ("Perimeter Intruder Test (intruder_test.mp4)", PROJECT_ROOT / "datasets" / "intruder_test.mp4"),
        ]
        self.sample_map = {}
        values = []
        for title, path in samples:
            if path.is_file():
                values.append(title)
                self.sample_map[title] = path

        if values:
            self.sample_combo.configure(values=values)

    # ------------------- VIDEO LOADING & UPLOAD -------------------
    def on_upload_video(self):
        filetypes = [
            ("CCTV Video Files", "*.mp4 *.avi *.mov"),
            ("MP4 Video", "*.mp4"),
            ("AVI Video", "*.avi"),
            ("MOV Video", "*.mov"),
            ("All Files", "*.*"),
        ]
        chosen = filedialog.askopenfilename(title="Select CCTV Surveillance Video", filetypes=filetypes)
        if chosen:
            self.load_video(Path(chosen), is_user_upload=True)

    def on_sample_selected(self, event=None):
        selected = self.sample_combo.get()
        if selected in self.sample_map:
            self.load_video(self.sample_map[selected], is_user_upload=False)

    def load_video(self, path: Path, is_user_upload: bool = True):
        if not path.is_file():
            messagebox.showerror("Error", f"Video file not found:\n{path}")
            return

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            messagebox.showerror("Error", f"Failed to open video file:\n{path}")
            return

        ret, frame0 = cap.read()
        if not ret or frame0 is None:
            messagebox.showerror("Error", f"Failed to decode the first frame of:\n{path.name}")
            cap.release()
            return

        self.video_path = path
        self.first_frame_bgr = frame0
        self.orig_h, self.orig_w = frame0.shape[:2]
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = self.total_frames / self.fps if self.fps > 0 else 0
        cap.release()

        # Update Video Info Label
        dur_m, dur_s = int(duration_sec // 60), int(duration_sec % 60)
        self.lbl_video_info.config(
            text=(
                f"File: {path.name}\n"
                f"Resolution: {self.orig_w}x{self.orig_h} | FPS: {self.fps:.1f}\n"
                f"Duration: {dur_m:02d}:{dur_s:02d} ({self.total_frames} frames)\n"
                f"Source: {'User Upload' if is_user_upload else 'Sample Library'}"
            ),
            fg=self.success_color,
        )

        # Clear old drawn zones and table when a new video is uploaded
        self.saved_zones = []
        self.current_poly_pts = []
        self.canvas_img_id = None
        self._update_zones_listbox()
        self._clear_incident_table()
        self.live_alert_banner.config(
            text="● ALL RESTRICTED ZONES & AREAS SECURE — NO ACTIVE ALERTS",
            bg="#1a2e22",
            fg=self.success_color,
        )
        self.status_badge.config(
            text="● READY",
            fg=self.success_color,
            bg="#1a2e22",
        )

        # Switch to Monitor Tab
        self.notebook.select(self.tab_monitor)
        self._redraw_canvas()
        self.status_text.config(text=f"Loaded {path.name}. Ready to start detection (or draw polygon for restricted zones).")

    def _on_feature_toggle(self):
        feat_intrusion = self.chk_feat_intrusion.get()
        feat_vehicle = self.chk_feat_vehicle.get()
        feat_bag = self.chk_feat_bag.get()
        feat_fight = self.chk_feat_fight.get()
        if not feat_intrusion and (feat_vehicle or feat_bag or feat_fight):
            active_names = []
            if feat_bag:
                active_names.append("Unattended-Bag Detection")
            if feat_vehicle:
                active_names.append("Vehicle Tracking")
            if feat_fight:
                active_names.append("Physical Altercation Detection")
            self.status_text.config(text=f"Restricted-Zone Intrusion disabled. {', '.join(active_names)} will run full-frame without requiring polygons.")
        elif feat_intrusion:
            self.status_text.config(text="Restricted-Zone Intrusion enabled. Draw polygon zone on video frame or use saved zones.")

    def _on_fight_threshold_change(self, value: str):
        self.lbl_fight_threshold_value.config(text=f"{float(value):.0%}")

    # ------------------- INTERACTIVE POLYGON DRAWING -------------------
    def on_canvas_click(self, event):
        if self.is_processing:
            return  # Disable drawing during live playback

        if self.first_frame_bgr is None:
            messagebox.showinfo("Upload Video First", "Please upload or select a video to display its first frame.")
            return

        cx, cy = event.x, event.y
        if (
            self.offset_x <= cx <= self.offset_x + self.disp_w
            and self.offset_y <= cy <= self.offset_y + self.disp_h
        ):
            self.current_poly_pts.append((cx, cy))
            self._redraw_canvas()

    def on_canvas_right_click(self, event):
        if self.is_processing:
            return
        if self.current_poly_pts:
            self.current_poly_pts.pop()
            self._redraw_canvas()

    def on_canvas_motion(self, event):
        if self.is_processing:
            return
        if self.current_poly_pts and self.first_frame_bgr is not None:
            self.temp_mouse_pos = (event.x, event.y)
            self._redraw_canvas(rubberband=True)

    def on_reset_current_points(self):
        if self.is_processing:
            return
        self.current_poly_pts = []
        self.temp_mouse_pos = None
        self._redraw_canvas()
        self.status_text.config(text="Cleared current in-progress points.")

    def on_save_zone(self):
        if self.is_processing:
            return
        if len(self.current_poly_pts) < 3:
            messagebox.showwarning("Incomplete Polygon", "A restricted-zone polygon requires at least 3 points.\nClick on the frame to place points.")
            return

        # Calculate normalized coordinates relative to native video resolution
        norm_points = []
        for cx, cy in self.current_poly_pts:
            nx = (cx - self.offset_x) / float(self.disp_w)
            ny = (cy - self.offset_y) / float(self.disp_h)
            norm_points.append([round(max(0.0, min(1.0, nx)), 4), round(max(0.0, min(1.0, ny)), 4)])

        z_name = self.zone_name_entry.get().strip() or f"Restricted Zone {len(self.saved_zones) + 1}"
        z_classes = set()
        if self.chk_person_var.get():
            z_classes.add("person")
        if self.chk_vehicle_var.get():
            z_classes.update(["car", "motorcycle", "bus", "truck"])

        zone_data = {
            "id": f"zone_{len(self.saved_zones) + 1}_{int(time.time())}",
            "name": z_name,
            "points": norm_points,
            "classes": sorted(list(z_classes)),
            "enabled": True,
            "zone_type": "restricted",
        }
        self.saved_zones.append(zone_data)
        self.current_poly_pts = []
        self.temp_mouse_pos = None

        self.zone_name_entry.delete(0, tk.END)
        self.zone_name_entry.insert(0, f"Restricted Zone {len(self.saved_zones) + 1}")

        self._update_zones_listbox()
        self._redraw_canvas()
        self.status_text.config(text=f"Saved '{z_name}' ({len(norm_points)} vertices). Ready to draw more or Start Detection.")

    def on_delete_selected_zone(self):
        if self.is_processing:
            return
        sel = self.zones_listbox.curselection()
        if sel:
            idx = sel[0]
            del self.saved_zones[idx]
            self._update_zones_listbox()
            self._redraw_canvas()

    def on_clear_all_zones(self):
        if self.is_processing:
            return
        self.saved_zones = []
        self.current_poly_pts = []
        self._update_zones_listbox()
        self._redraw_canvas()
        self.status_text.config(text="All restricted zones cleared.")

    def _update_zones_listbox(self):
        self.zones_listbox.delete(0, tk.END)
        for idx, z in enumerate(self.saved_zones):
            classes_str = ", ".join(z.get("classes", []))
            self.zones_listbox.insert(tk.END, f"#{idx + 1} {z['name']} ({len(z['points'])} pts) [{classes_str}]")
        self.lbl_active_zones_count.config(text=f"Active Zones: {len(self.saved_zones)}")

    def _redraw_canvas(self, rubberband: bool = False):
        if self.is_processing or self.first_frame_bgr is None:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 50 or canvas_h < 50:
            return

        scale = min(canvas_w / float(self.orig_w), canvas_h / float(self.orig_h))
        self.disp_w = max(1, int(self.orig_w * scale))
        self.disp_h = max(1, int(self.orig_h * scale))
        self.offset_x = (canvas_w - self.disp_w) // 2
        self.offset_y = (canvas_h - self.disp_h) // 2

        rgb_frame = cv2.cvtColor(self.first_frame_bgr, cv2.COLOR_BGR2RGB)
        resized_pil = Image.fromarray(rgb_frame).resize((self.disp_w, self.disp_h), Image.Resampling.BILINEAR)

        self.photo_img = ImageTk.PhotoImage(resized_pil)
        self.canvas.delete("all")
        self.canvas_img_id = self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.photo_img)

        # 1. Draw Already Saved Zones in Yellow
        for idx, z in enumerate(self.saved_zones):
            pts_canvas = []
            for nx, ny in z["points"]:
                cx = int(self.offset_x + nx * self.disp_w)
                cy = int(self.offset_y + ny * self.disp_h)
                pts_canvas.append((cx, cy))

            if len(pts_canvas) >= 3:
                flat_pts = [coord for pt in pts_canvas for coord in pt]
                self.canvas.create_polygon(
                    flat_pts,
                    outline="#FFFF00",
                    width=2,
                    fill="#FFFF00",
                    stipple="gray25",
                )
                top_pt = min(pts_canvas, key=lambda p: p[1])
                self.canvas.create_text(
                    top_pt[0],
                    max(15, top_pt[1] - 12),
                    text=f"ZONE: {z['name']} [RESTRICTED]",
                    fill="#FFFF00",
                    font=("Segoe UI", 9, "bold"),
                    anchor="w",
                )

        # 2. Draw In-Progress Polygon Points & Lines
        for idx, (px, py) in enumerate(self.current_poly_pts):
            self.canvas.create_oval(px - 5, py - 5, px + 5, py + 5, fill="#e06c75", outline="#FFFF00", width=2)
            self.canvas.create_text(px + 8, py - 8, text=str(idx + 1), fill="#ffffff", font=("Segoe UI", 8, "bold"))

        if len(self.current_poly_pts) > 1:
            for i in range(len(self.current_poly_pts) - 1):
                p1 = self.current_poly_pts[i]
                p2 = self.current_poly_pts[i + 1]
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#FFFF00", width=2)

        if rubberband and self.current_poly_pts and self.temp_mouse_pos:
            last = self.current_poly_pts[-1]
            mx, my = self.temp_mouse_pos
            self.canvas.create_line(last[0], last[1], mx, my, fill="#e5c07b", dash=(4, 4), width=2)

    # ------------------- DETECTION EXECUTION & REAL-TIME ANIMATION -------------------
    def on_start_detection(self):
        if self.video_path is None or not self.video_path.is_file():
            messagebox.showwarning("No Video Selected", "Please upload or select a CCTV video file before starting detection.")
            return

        feat_intrusion = self.chk_feat_intrusion.get()
        feat_vehicle = self.chk_feat_vehicle.get()
        feat_bag = self.chk_feat_bag.get()
        feat_fight = self.chk_feat_fight.get()

        if not (feat_intrusion or feat_vehicle or feat_bag or feat_fight):
            messagebox.showwarning(
                "No Feature Selected",
                "Please select at least one AI monitoring feature:\n"
                "• Restricted-Zone Intrusion\n"
                "• Vehicle Detection & Tracking\n"
                "• Unattended-Bag Detection\n"
                "• Physical Altercation Detection"
            )
            return

        runtime_zones: list[RestrictedZone] = []
        if feat_intrusion:
            if not self.saved_zones:
                if len(self.current_poly_pts) >= 3:
                    self.on_save_zone()
                else:
                    messagebox.showwarning(
                        "No Restricted Zone Defined",
                        "Restricted-Zone Intrusion Detection is enabled.\n"
                        "Please draw at least one zone polygon directly on the video frame "
                        "and click 'Save Zone', or uncheck it to run the selected full-frame "
                        "features without polygons."
                    )
                    return

            for z in self.saved_zones:
                rz = RestrictedZone(
                    zone_id=z["id"],
                    name=z["name"],
                    raw_points=z["points"],
                    classes=set(z.get("classes", ["person", "car", "motorcycle", "bus", "truck"])),
                    enabled=True,
                    is_normalized=True,
                    zone_type=z.get("zone_type", "restricted"),
                )
                runtime_zones.append(rz)
        else:
            # Full-frame features do not require a restricted-zone polygon.
            self.current_poly_pts = []

        out_dir = PROJECT_ROOT / "runs" / "user_uploads"
        out_dir.mkdir(parents=True, exist_ok=True)
        screens_dir = out_dir / "screenshots"
        screens_dir.mkdir(parents=True, exist_ok=True)

        video_stem = self.video_path.stem
        out_video_path = out_dir / f"{video_stem}_annotated.mp4"

        m_name = self.model_combo.get()
        model_path = PROJECT_ROOT / m_name
        if not model_path.is_file():
            model_path = PROJECT_ROOT / "yolov8s.pt"
            if not model_path.is_file():
                model_path = PROJECT_ROOT / "yolov8n.pt"

        conf = float(self.scale_conf.get())
        confirm_frames = int(self.spin_confirm.get())
        sched_mode = self.sched_combo.get()

        try:
            stat_time = float(self.spin_stationary.get())
        except Exception:
            stat_time = 15.0

        try:
            p_dist = float(self.spin_person_dist.get())
        except Exception:
            p_dist = 150.0

        self.is_processing = True
        self.stop_event.clear()
        self.canvas_img_id = None
        self.last_ui_draw_time = 0.0

        # Drain and clear UI queue
        while not self.ui_queue.empty():
            try:
                self.ui_queue.get_nowait()
            except queue.Empty:
                break

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.progress_bar.config(value=0)
        self.status_badge.config(text="● RUNNING PIPELINE", fg=self.alert_color, bg="#331c1e")

        active_str = []
        if feat_intrusion:
            active_str.append(f"Intrusion ({len(runtime_zones)} zones)")
        if feat_vehicle:
            active_str.append("Vehicles")
        if feat_bag:
            active_str.append("Unattended Bags (Auto)")
        if feat_fight:
            active_str.append("Physical Altercation (Possible)")
        self.status_text.config(text=f"Processing {self.video_path.name} [{', '.join(active_str)}]...")

        self._clear_incident_table()

        # Switch to Monitor Tab so user sees moving video & live logs
        self.notebook.select(self.tab_monitor)

        # Start thread-safe UI poll loop on main thread
        self.root.after(1, self._poll_ui_queue)

        thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(
                feat_intrusion, feat_vehicle, feat_bag, feat_fight,
                float(self.fight_threshold_var.get()),
                runtime_zones, model_path, out_video_path, screens_dir,
                conf, confirm_frames, sched_mode, stat_time, p_dist
            ),
            daemon=True,
        )
        thread.start()

    def _poll_ui_queue(self):
        try:
            msg_type, payload = self.ui_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            if msg_type == "COMPLETED":
                results, out_video_path = payload
                self._on_pipeline_completed(results, out_video_path)
                return
            if msg_type == "ERROR":
                self._on_pipeline_error(payload)
                return
            if msg_type == "FRAME":
                latest_frame_data = payload
                self._update_live_ui(
                    latest_frame_data["frame_idx"],
                    latest_frame_data["total_frames"],
                    latest_frame_data["fps"],
                    latest_frame_data["timestamp"],
                    latest_frame_data["status"],
                    latest_frame_data["annotated_frame"],
                    latest_frame_data["incident_history"],
                )

        if self.is_processing or not self.ui_queue.empty():
            poll_interval_ms = max(1, int(1000 / max(float(self.fps), 1.0)))
            self.root.after(poll_interval_ms, self._poll_ui_queue)

    def on_stop_detection(self):
        if self.is_processing:
            self.stop_event.set()
            self.status_text.config(text="Stopping pipeline...")

    def _run_pipeline_thread(
        self,
        feat_intrusion: bool,
        feat_vehicle: bool,
        feat_bag: bool,
        feat_fight: bool,
        fight_threshold: float,
        zones: list[RestrictedZone],
        model_path: Path,
        out_video_path: Path,
        screens_dir: Path,
        conf: float,
        confirm_frames: int,
        sched_mode: str,
        stat_time: float,
        p_dist: float,
    ):
        try:
            print(f"\n=======================================================")
            print(f"[Smart CCTV App] Initializing integrated pipeline on: {self.video_path.name}")
            print(f"[Smart CCTV App] Model: {model_path.name} | Conf: {conf:.2f}")
            print(
                f"[Smart CCTV App] Features: Intrusion={feat_intrusion}, "
                f"Vehicles={feat_vehicle}, UnattendedBag={feat_bag}, "
                f"PossibleFight={feat_fight}"
            )
            print(f"=======================================================\n")

            pipeline = IntegratedPipeline(
                yolo_model_path=model_path,
                enable_intrusion=feat_intrusion,
                enable_vehicle_tracking=feat_vehicle,
                enable_unattended_bag=feat_bag,
                zones=zones,
                confidence=conf,
                bag_confidence=0.25,
                confirm_frames=confirm_frames,
                stationary_time=stat_time,
                person_distance=p_dist,
                stationary_distance=25.0,
                alert_cooldown=30.0,
                sched_mode=sched_mode,
                camera_name=f"{self.video_path.stem.replace('_', ' ').title()} CCTV",
                enable_fight_detection=feat_fight,
                fight_confidence_threshold=fight_threshold,
            )

            last_terminal_print_time = [0.0]

            def progress_cb(data):
                f_idx = data["frame_idx"]
                tot = data["total_frames"]
                cur_fps = data["fps"]
                cur_ts = data["timestamp"]
                status = data["status"]
                annotated_frame = data["annotated_frame"]

                now = time.time()
                has_new_alerts = len(status.get("notifications", [])) > 0 or status.get("all_active_alerts", 0) > 0
                incidents_snapshot = data["incident_history"]

                # Terminal frame-by-frame playback logging (every 1s or on alert)
                mins = int(cur_ts) // 60
                secs = int(cur_ts) % 60
                time_disp = f"{mins:02d}:{secs:02d}"
                if has_new_alerts and (now - last_terminal_print_time[0] >= 0.5):
                    last_terminal_print_time[0] = now
                    confirmed = status.get("confirmed_intruders", [])
                    bags = status.get("unattended_bags", [])
                    if confirmed:
                        intr_strs = [f"{i.class_name.capitalize()} #{i.track_id} → '{i.zone_name}'" for i in confirmed]
                        print(f"  [VIDEO {time_disp}] Frame {f_idx:5d} | ⚠ ALERT: {', '.join(intr_strs)}", flush=True)
                    if bags:
                        bag_strs = [
                            f"{b.class_name.capitalize()} (Person #{b.associated_person_id})"
                            if b.associated_person_id is not None
                            else f"{b.class_name.capitalize()} (Owner unknown)"
                            for b in bags
                        ]
                        print(f"  [VIDEO {time_disp}] Frame {f_idx:5d} | ⚠ UNATTENDED BAG: {', '.join(bag_strs)}", flush=True)
                elif now - last_terminal_print_time[0] >= 5.0:
                    last_terminal_print_time[0] = now
                    print(f"  [VIDEO {time_disp}] Frame {f_idx:5d}/{tot} | {cur_fps:.1f} FPS | Incidents: {len(incidents_snapshot)} | Status: Normal", flush=True)

                payload = {
                    "frame_idx": f_idx,
                    "total_frames": tot,
                    "fps": cur_fps,
                    "timestamp": cur_ts,
                    "status": status,
                    "annotated_frame": annotated_frame,
                    "incident_history": incidents_snapshot,
                }
                self.ui_queue.put(("FRAME", payload))

            results = process_integrated_stream(
                video_source=str(self.video_path),
                pipeline=pipeline,
                output_video=out_video_path,
                save_screenshots=screens_dir,
                progress_callback=progress_cb,
                stop_event=self.stop_event,
                realtime_playback=True,
            )

            self.ui_queue.put(("COMPLETED", (results, out_video_path)))

        except Exception as error:
            print(f"[Smart CCTV App] Integrated pipeline error: {error}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            self.ui_queue.put(("ERROR", str(error)))

    def _update_live_ui(
        self,
        frame_idx: int,
        total_frames: int,
        cur_fps: float,
        timestamp: float,
        status: dict,
        frame_bgr: np.ndarray,
        incident_history: list[dict],
    ):
        # 1. Update Progress Bar & Label
        pct = (frame_idx / float(total_frames)) * 100.0 if total_frames > 0 else 0.0
        self.progress_bar.config(value=pct)

        alert_count = status.get("all_active_alerts", 0)
        banner_text = status.get("banner_text")
        banner_level = status.get("banner_level", "NORMAL")
        active_bags = status.get("unattended_bags", [])

        # 2. Live Security Alert Banner
        if (
            status.get("possible_fight")
            and not status.get("confirmed_intruders")
            and not active_bags
        ):
            self.live_alert_banner.config(
                text=banner_text or "⚠ POSSIBLE PHYSICAL ALTERCATION — HUMAN REVIEW REQUIRED",
                bg="#8a1c24",
                fg="#ffffff",
            )
            self.status_badge.config(
                text="⚠ POSSIBLE FIGHT",
                fg="#ff6b6b",
                bg="#3d1416",
            )
        elif banner_level == "CRITICAL" or alert_count > 0:
            self.live_alert_banner.config(
                text=banner_text or f"🚨 CRITICAL ALERT ACTIVE ({alert_count} active)",
                bg="#8a1c24",
                fg="#ffffff",
            )
            self.status_badge.config(text=f"🚨 {alert_count} ACTIVE ALERTS", fg="#ff6b6b", bg="#3d1416")
        else:
            self.live_alert_banner.config(
                text="● ALL RESTRICTED ZONES & AREAS SECURE — NO ACTIVE ALERTS",
                bg="#1a2e22",
                fg=self.success_color,
            )
            self.status_badge.config(text="● RUNNING PIPELINE", fg=self.accent_color, bg="#1a2333")

        self.lbl_progress.config(
            text=f"Frame {frame_idx}/{total_frames} ({pct:.1f}%) | {cur_fps:.1f} FPS | t={timestamp:.2f}s"
        )

        # 3. Live Moving Video Screen Rendering
        if frame_bgr is not None:
            self._display_frame_on_canvas(frame_bgr)

        # 4. Live KPI Cards
        self._set_current_incidents(incident_history)
        self._refresh_incident_dashboard(
            active_alert_count=alert_count,
            fps=cur_fps,
        )

        # 5. Dedicated Security Notification Panel (Below Video)
        unattended_bags = status.get("unattended_bags", [])
        active_bag_alerts = [b for b in unattended_bags if getattr(b, "alert_active", False)]
        notifications = status.get("notifications", [])

        if status.get("possible_fight"):
            probability = status.get("fight_probability") or 0.0
            fight_time = status.get("fight_time_str") or "--:--:--"
            notif_msg = (
                f"⚠ [POSSIBLE PHYSICAL ALTERCATION] {probability:.0%} fight "
                f"probability at {fight_time} | HUMAN REVIEW REQUIRED — "
                "individuals are not identified."
            )
            self.lbl_notif_text.config(text=notif_msg, fg="#ff6b6b", bg="#3d1416")
            self.lbl_notif_title.config(fg="#ff6b6b", bg="#3d1416")
            self.notif_panel.config(bg="#3d1416")
        elif active_bag_alerts:
            first_bag = active_bag_alerts[0]
            dur = max(0.0, timestamp - getattr(first_bag, "stationary_since", timestamp))
            owner_info = f"Person #{first_bag.associated_person_id}" if getattr(first_bag, "associated_person_id", None) is not None else "Away / Unknown"
            notif_msg = (
                f"🚨 [UNATTENDED BAG ALERT] {first_bag.class_name.capitalize()} "
                f"stationary for {dur:.1f}s (Owner: {owner_info}) at {status.get('time_str')} | Status: ACTIVE"
            )
            self.lbl_notif_text.config(text=notif_msg, fg="#ff6b6b", bg="#3d1416")
            self.lbl_notif_title.config(fg="#ff6b6b", bg="#3d1416")
            self.notif_panel.config(bg="#3d1416")
        elif status.get("confirmed_intruders"):
            first_intr = status["confirmed_intruders"][0]
            notif_msg = (
                f"🚨 [RESTRICTED ZONE INTRUSION] {first_intr.class_name.capitalize()} #{first_intr.track_id} "
                f"in '{first_intr.zone_name}' at {status.get('time_str')} | Status: ACTIVE"
            )
            self.lbl_notif_text.config(text=notif_msg, fg="#ff6b6b", bg="#3d1416")
            self.lbl_notif_title.config(fg="#ff6b6b", bg="#3d1416")
            self.notif_panel.config(bg="#3d1416")
        elif notifications:
            latest = notifications[-1]
            lvl = latest.get("level", "INFO")
            is_crit = lvl == "CRITICAL"
            fg_col = "#ff6b6b" if is_crit else self.success_color
            bg_col = "#3d1416" if is_crit else "#1a2e22"
            self.lbl_notif_text.config(text=f"[{lvl}] {latest.get('message', '')}", fg=fg_col, bg=bg_col)
            self.lbl_notif_title.config(fg=fg_col, bg=bg_col)
            self.notif_panel.config(bg=bg_col)
        elif status.get("fight_probability") is not None:
            probability = status["fight_probability"]
            fight_time = status.get("fight_time_str") or "--:--:--"
            self.lbl_notif_text.config(
                text=(
                    f"Physical altercation model: {probability:.0%} at {fight_time} "
                    f"(alert threshold {self.fight_threshold_var.get():.0%}). "
                    "Below threshold; not a confirmed fight."
                ),
                fg=self.text_color,
                bg="#1a1d24",
            )
            self.lbl_notif_title.config(fg="#61afef", bg="#1a1d24")
            self.notif_panel.config(bg="#1a1d24")
        else:
            self.lbl_notif_text.config(text="● All restricted zones & areas secure — No active alerts.", fg=self.success_color, bg="#1a1d24")
            self.lbl_notif_title.config(fg="#61afef", bg="#1a1d24")
            self.notif_panel.config(bg="#1a1d24")

    def _display_frame_on_canvas(self, frame_bgr: np.ndarray):
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 50 or canvas_h < 50:
            return

        h, w = frame_bgr.shape[:2]
        scale = min(canvas_w / float(w), canvas_h / float(h))
        disp_w = max(1, int(w * scale))
        disp_h = max(1, int(h * scale))
        offset_x = (canvas_w - disp_w) // 2
        offset_y = (canvas_h - disp_h) // 2

        self.disp_w = disp_w
        self.disp_h = disp_h
        self.offset_x = offset_x
        self.offset_y = offset_y

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        self.live_photo = ImageTk.PhotoImage(pil_img)

        if self.canvas_img_id is None:
            self.canvas.delete("all")
            self.canvas_img_id = self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.live_photo)
        else:
            self.canvas.coords(self.canvas_img_id, offset_x, offset_y)
            self.canvas.itemconfig(self.canvas_img_id, image=self.live_photo)

    def _sync_incident_table(
        self,
        live_incidents: list[dict],
        audit_incidents: list[dict] | None = None,
    ):
        def sync_tree(tree, incidents: list[dict], live: bool = False) -> None:
            existing_items = tree.get_children()
            for i, inc in enumerate(incidents):
                dur_str = (
                    f"{inc['duration_seconds']}s"
                    if inc.get("duration_seconds") is not None
                    else "--"
                )
                displayed_track = f"#{inc.get('track_id')}"
                if live and inc.get("type") == "UNATTENDED_BAG":
                    owner_id = inc.get("owner_id")
                    displayed_track = f"Person #{owner_id}" if owner_id is not None else "Unknown"
                vals = (
                    inc.get("incident_id", i + 1),
                    inc.get("type", "INTRUSION"),
                    "N/A" if inc.get("type") == "POSSIBLE_FIGHT" else displayed_track,
                    inc.get("class_name", "person").capitalize(),
                    inc.get("zone_name", "Zone"),
                    inc.get("entry_time", "--"),
                    inc.get("exit_time") or "--",
                    dur_str,
                    inc.get("status", "ACTIVE"),
                )
                if i < len(existing_items):
                    tree.item(existing_items[i], values=vals)
                else:
                    tree.insert("", "end", values=vals)
            for item_id in existing_items[len(incidents):]:
                tree.delete(item_id)

        sync_tree(self.tree, live_incidents, live=True)
        sync_tree(
            self.tree_audit,
            audit_incidents if audit_incidents is not None else live_incidents,
        )

    def _clear_incident_table(self):
        self._incident_table_signature = None
        self._current_incident_history = []
        self._refresh_incident_dashboard(active_alert_count=0)
        self.kpi_fps.val_lbl.config(text="-- FPS")
        if hasattr(self, "lbl_notif_text"):
            self.lbl_notif_text.config(
                text="System ready. Draw restricted zone or select features to start monitoring.",
                fg=self.text_bright,
                bg="#1a1d24",
            )
            self.lbl_notif_title.config(fg="#61afef", bg="#1a1d24")
            self.notif_panel.config(bg="#1a1d24")

    def _on_pipeline_completed(self, results: dict, out_video_path: Path):
        self.is_processing = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_badge.config(text="● ANALYSIS COMPLETE", fg=self.success_color, bg="#1d2d22")
        self.status_text.config(text=f"Analysis completed! Processed {results.get('total_frames', 0)} frames in {results.get('elapsed_seconds', 0):.1f}s.")

        self.last_results = results
        self.annotated_video_path = out_video_path
        self.saved_screenshots = results.get("saved_screenshots", [])

        self.btn_play_video.config(state="normal")
        self.btn_open_folder.config(state="normal")
        self.btn_export_log.config(state="normal")

        # Populate Screenshots Gallery on Tab 2
        self._populate_screenshots_gallery()

        run_total_incidents = results.get("total_incidents", 0)
        run_person_incidents = results.get("person_incidents", 0)
        run_vehicle_incidents = results.get("vehicle_incidents", 0)
        run_bag_incidents = results.get("unattended_bag_incidents", 0)
        final_history = results.get("incident_history", [])
        self._set_current_incidents(final_history)
        active_alerts = sum(
            1 for incident in final_history
            if incident.get("status", "").upper() == "ACTIVE"
        )

        # --- UPDATE KPI CARDS with saved and current incident counts ---
        self._refresh_incident_dashboard(
            active_alert_count=active_alerts,
            fps=results.get("average_fps", 0.0),
        )
        if active_alerts:
            self.status_badge.config(
                text=f"🚨 {active_alerts} ACTIVE ALERTS",
                fg="#ff6b6b",
                bg="#3d1416",
            )
        # Update Notification Panel on completion
        if hasattr(self, "lbl_notif_text"):
            completion_message = (
                f"● Detection complete: {run_total_incidents} incident(s) logged; "
                f"{active_alerts} alert(s) remain active at the end of the video."
                if active_alerts
                else f"● Detection Complete: {run_total_incidents} incident(s) logged "
                f"(Person: {run_person_incidents}, Vehicle: {run_vehicle_incidents}, "
                f"Bags: {run_bag_incidents})."
            )
            completion_color = "#ff6b6b" if active_alerts else self.success_color
            completion_bg = "#3d1416" if active_alerts else "#1a2e22"
            self.lbl_notif_text.config(
                text=completion_message,
                fg=completion_color,
                bg=completion_bg,
            )
            self.lbl_notif_title.config(fg=completion_color, bg=completion_bg)
            self.notif_panel.config(bg=completion_bg)

        messagebox.showinfo(
            "Detection Complete",
            f"Video processing finished!\n\n"
            f"• Total Incidents: {run_total_incidents}\n"
            f"• Person Intrusions: {run_person_incidents}\n"
            f"• Vehicle Intrusions: {run_vehicle_incidents}\n"
            f"• Unattended Bags: {run_bag_incidents}\n"
            f"• Output Video: {out_video_path.name}\n"
            f"• Event Screenshots: {len(self.saved_screenshots)} captured\n\n"
            f"Click 'Play Annotated Video' to watch the full playback."
        )

    def _populate_screenshots_gallery(self):
        for widget in self.gallery_inner.winfo_children():
            widget.destroy()
        self.thumb_refs = []

        if not self.saved_screenshots:
            lbl = tk.Label(self.gallery_inner, text="No intrusion screenshots captured.", fg=self.text_color, bg=self.card_bg)
            lbl.pack(padx=20, pady=20)
            return

        for p_str in self.saved_screenshots:
            p = Path(p_str)
            if not p.is_file():
                continue
            try:
                img_cv = cv2.imread(str(p))
                if img_cv is None:
                    continue
                rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                th_pil = Image.fromarray(rgb).resize((130, 80), Image.Resampling.BILINEAR)
                th_photo = ImageTk.PhotoImage(th_pil)
                self.thumb_refs.append(th_photo)

                card = tk.Frame(self.gallery_inner, bg="#1e2227", padx=4, pady=4, relief="ridge", cursor="hand2")
                card.pack(side="left", padx=6, pady=4)

                btn = tk.Label(card, image=th_photo, bg="#1e2227")
                btn.pack()
                btn.bind("<Button-1>", lambda e, path=p: os.startfile(str(path)))

                name_lbl = tk.Label(card, text=p.name[:18], font=("Segoe UI", 7), fg=self.text_color, bg="#1e2227")
                name_lbl.pack()
            except Exception:
                pass

    def _on_pipeline_error(self, err_msg: str):
        self.is_processing = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_badge.config(text="● PIPELINE ERROR", fg=self.alert_color, bg="#331c1e")
        self.status_text.config(text=f"Error: {err_msg}")
        messagebox.showerror("Pipeline Execution Error", f"An error occurred while processing the video:\n{err_msg}")

    # ------------------- RESULTS ACTIONS -------------------
    def on_play_annotated_video(self):
        if self.annotated_video_path and self.annotated_video_path.is_file():
            try:
                os.startfile(str(self.annotated_video_path))
            except Exception:
                threading.Thread(target=self._play_opencv, args=(self.annotated_video_path,), daemon=True).start()
        else:
            messagebox.showwarning("File Missing", "Annotated video not found.")

    def _play_opencv(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        win = f"Annotated Playback - {video_path.name} (Press Q to exit)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(win, frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyWindow(win)

    def on_open_output_folder(self):
        if self.annotated_video_path:
            folder = self.annotated_video_path.parent
            try:
                os.startfile(str(folder))
            except Exception:
                messagebox.showinfo("Output Folder", f"Output files are located in:\n{folder}")

    def on_export_incident_report(self):
        if not self.last_results or not self.last_results.get("incident_history"):
            messagebox.showinfo("No Data", "No incident history to export.")
            return

        save_target = filedialog.asksaveasfilename(
            title="Save Incident Report",
            defaultextension=".csv",
            filetypes=[("CSV Spreadsheet", "*.csv"), ("JSON File", "*.json")],
        )
        if save_target:
            p = Path(save_target)
            history = self.last_results["incident_history"]
            if p.suffix.lower() == ".json":
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)
            else:
                with open(p, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                    writer.writeheader()
                    writer.writerows(history)
            messagebox.showinfo("Export Successful", f"Incident audit report saved to:\n{p.name}")


def main():
    root = tk.Tk()
    app = SmartCCTVApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
