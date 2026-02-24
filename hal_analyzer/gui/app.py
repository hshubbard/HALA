"""
HAL Behavior Analyzer - GUI Frontend (Tkinter)

The GUI calls the core AnalysisEngine directly (same as the CLI does).
All analysis logic stays in the engine - the GUI just invokes it and
renders the results.

Layout
------
+----------------------------------------------------------------------+
|  [Analyze Trace]  [Load Result JSON]  source label  | Min sev: [v]  |
|  Arch: [______]                                                       |
+-------------------------+--------------------------------------------+
|  Call Sequence          |  Findings                                   |
|  (scrollable table)     |  [All][Contention][Fragility][Performance]  |
|  - flagged rows colour  |  (scrollable list)                          |
|  - click = jump/detail  |                                             |
|                         |  Detail panel (explanation + suggestion)    |
+-------------------------+--------------------------------------------+
|  Resource Lifetime Timeline (scrollable canvas)                       |
+----------------------------------------------------------------------+
|  Status bar                                                           |
+----------------------------------------------------------------------+

Usage
-----
  python launch_gui.py
  python launch_gui.py examples/trace_pingpong.json   # auto-analyze on open
  python launch_gui.py examples/result_pingpong.json  # load saved result
"""

from __future__ import annotations

import json
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

# Ensure package root is on path when run as a script
_here = Path(__file__).resolve().parent.parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from hal_analyzer.core.engine import AnalysisEngine, AnalysisResult

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────────────────────────────────────

SEV_COLOURS = {
    "critical" : "#FF4444",
    "high"     : "#FF8800",
    "medium"   : "#FFCC00",
    "low"      : "#66BBFF",
    "info"     : "#AAAAAA",
}

RES_KIND_COLOURS = {
    "buffer"    : "#4A90D9",
    "texture"   : "#7B68EE",
    "fence"     : "#50C878",
    "semaphore" : "#FFA07A",
    "queue"     : "#FFD700",
    "pipeline"  : "#20B2AA",
    "descriptor": "#DA70D6",
    "unknown"   : "#888888",
}

DARK_BG  = "#1e1e1e"
DARK_FG  = "#d4d4d4"
PANEL_BG = "#252526"
ACCENT   = "#007ACC"
ROW_EVEN = "#2d2d2d"
ROW_ODD  = "#252526"
ROW_FLAG = "#3d2000"


# ──────────────────────────────────────────────────────────────────────────────
# Display model — wraps the result dict for view-friendly access
# ──────────────────────────────────────────────────────────────────────────────

class DisplayModel:
    def __init__(self, data: Dict[str, Any]) -> None:
        self.data      = data
        self.findings  : List[Dict[str, Any]] = data.get("findings", [])
        self.trace     : Dict[str, Any]       = data.get("trace", {})
        self.nodes     : List[Dict[str, Any]] = self.trace.get("nodes", [])
        self.resources : Dict[str, Any]       = self.trace.get("resources", {})
        self.lifetimes : Dict[str, Any]       = self.trace.get("lifetimes", {})
        self.summary   : Dict[str, Any]       = data.get("summary", {})
        self.source    : str                  = data.get("source_path", "(unknown)")
        self.arch      : str                  = data.get("arch_hint") or ""

        self._node_findings: Dict[int, List[Dict[str, Any]]] = {}
        for f in self.findings:
            for nidx in f.get("related_nodes", []):
                self._node_findings.setdefault(nidx, []).append(f)

    @classmethod
    def from_result(cls, result: AnalysisResult) -> "DisplayModel":
        return cls(result.to_dict())

    def findings_for_node(self, index: int) -> List[Dict[str, Any]]:
        return self._node_findings.get(index, [])

    def max_severity_for_node(self, index: int) -> Optional[str]:
        order = ["critical", "high", "medium", "low", "info"]
        best = None
        for f in self.findings_for_node(index):
            sev = f.get("severity", "info")
            if best is None or order.index(sev) < order.index(best):
                best = sev
        return best


# ──────────────────────────────────────────────────────────────────────────────
# Main application
# ──────────────────────────────────────────────────────────────────────────────

class HALAnalyzerApp(tk.Tk):

    def __init__(self, initial_file: Optional[str] = None) -> None:
        super().__init__()
        self.title("HAL Behavior Analyzer")
        self.geometry("1400x900")
        self.configure(bg=DARK_BG)
        self.minsize(900, 600)

        self._model: Optional[DisplayModel] = None
        self._engine = AnalysisEngine()

        self._build_ui()

        if initial_file:
            self._auto_open(initial_file)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_toolbar()
        self._build_main_pane()
        self._build_status_bar()
        self._apply_theme()

    def _apply_theme(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",        background=PANEL_BG, borderwidth=0)
        style.configure("TNotebook.Tab",    background=PANEL_BG, foreground=DARK_FG, padding=[8, 4])
        style.map("TNotebook.Tab",          background=[("selected", ACCENT)])
        style.configure("TFrame",           background=PANEL_BG)
        style.configure("TPanedwindow",     background=DARK_BG)
        style.configure("Treeview",         background=PANEL_BG, foreground=DARK_FG,
                         fieldbackground=PANEL_BG, rowheight=22)
        style.configure("Treeview.Heading", background=DARK_BG, foreground=DARK_FG)
        style.map("Treeview",               background=[("selected", ACCENT)])
        style.configure("TScrollbar",       background=PANEL_BG, troughcolor=DARK_BG)
        style.configure("TLabel",           background=PANEL_BG, foreground=DARK_FG)
        style.configure("TEntry",           fieldbackground="#3c3c3c", foreground=DARK_FG,
                         insertcolor=DARK_FG)

    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, bg=DARK_BG)
        bar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        btn_kw = dict(
            bg=ACCENT, fg="white", relief=tk.FLAT, padx=12, pady=4,
            font=("Segoe UI", 9), cursor="hand2", activebackground="#005f9e",
        )

        # Primary action — analyze a raw trace
        tk.Button(bar, text="Analyze Trace...",
                  command=self._open_trace_dialog, **btn_kw).pack(side=tk.LEFT, padx=(0, 4))

        # Secondary action — load a pre-computed result JSON
        tk.Button(bar, text="Load Result JSON...",
                  command=self._open_result_dialog,
                  bg="#3c3c3c", fg=DARK_FG, relief=tk.FLAT, padx=10, pady=4,
                  font=("Segoe UI", 9), cursor="hand2",
                  activebackground="#505050").pack(side=tk.LEFT, padx=(0, 12))

        # Arch hint entry
        tk.Label(bar, text="Arch:", bg=DARK_BG, fg=DARK_FG,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self._arch_var = tk.StringVar()
        arch_entry = ttk.Entry(bar, textvariable=self._arch_var, width=12)
        arch_entry.pack(side=tk.LEFT, padx=(2, 12))
        _tooltip(arch_entry, "Optional architecture hint (e.g. ampere, rdna3, arm-mali)")

        # Source label
        self._source_label = tk.Label(
            bar, text="No trace loaded", bg=DARK_BG, fg="#888888",
            font=("Consolas", 9)
        )
        self._source_label.pack(side=tk.LEFT, padx=4)

        # Severity filter (right-aligned)
        tk.Label(bar, text="Min severity:", bg=DARK_BG, fg=DARK_FG,
                 font=("Segoe UI", 9)).pack(side=tk.RIGHT, padx=(0, 4))
        self._sev_var = tk.StringVar(value="info")
        sev_menu = ttk.Combobox(bar, textvariable=self._sev_var, width=9,
                                 values=["info", "low", "medium", "high", "critical"],
                                 state="readonly")
        sev_menu.pack(side=tk.RIGHT, padx=(0, 6))
        sev_menu.bind("<<ComboboxSelected>>", lambda _: self._refresh_findings_panel())

    def _build_main_pane(self) -> None:
        top_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        top_pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        left_frame = ttk.Frame(top_pane)
        top_pane.add(left_frame, weight=2)
        self._build_call_sequence_panel(left_frame)

        right_frame = ttk.Frame(top_pane)
        top_pane.add(right_frame, weight=1)
        self._build_findings_panel(right_frame)

        self._build_lifetime_panel()

    def _build_call_sequence_panel(self, parent: tk.Widget) -> None:
        tk.Label(parent, text="Call Sequence", bg=PANEL_BG, fg=DARK_FG,
                 font=("Segoe UI Semibold", 10), anchor="w").pack(fill=tk.X, pady=(4, 0))

        cols = ("idx", "op", "category", "blocking", "queue", "flags")
        self._seq_tree = ttk.Treeview(parent, columns=cols, show="headings", selectmode="browse")
        widths = {"idx": 50, "op": 200, "category": 130, "blocking": 110, "queue": 90, "flags": 80}
        for c in cols:
            self._seq_tree.heading(c, text=c.title())
            self._seq_tree.column(c, width=widths[c], stretch=(c == "op"))

        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL,   command=self._seq_tree.yview)
        hsb = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self._seq_tree.xview)
        self._seq_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._seq_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)

        for sev, colour in SEV_COLOURS.items():
            self._seq_tree.tag_configure(f"sev_{sev}", background=ROW_FLAG, foreground=colour)
        self._seq_tree.tag_configure("even", background=ROW_EVEN)
        self._seq_tree.tag_configure("odd",  background=ROW_ODD)
        self._seq_tree.bind("<<TreeviewSelect>>", self._on_node_selected)

    def _build_findings_panel(self, parent: tk.Widget) -> None:
        tk.Label(parent, text="Findings", bg=PANEL_BG, fg=DARK_FG,
                 font=("Segoe UI Semibold", 10), anchor="w").pack(fill=tk.X, pady=(4, 0))

        self._findings_nb = ttk.Notebook(parent)
        self._findings_nb.pack(fill=tk.BOTH, expand=True)

        for label, key in [
            ("All",                 "all"),
            ("Contention",          "resource_contention"),
            ("Fragility",           "fragility"),
            ("Performance",         "performance"),
        ]:
            frame = ttk.Frame(self._findings_nb)
            self._findings_nb.add(frame, text=label)
            self._build_findings_list(frame, key)

        self._detail_text = tk.Text(
            parent, height=11, bg=DARK_BG, fg=DARK_FG,
            font=("Consolas", 9), wrap=tk.WORD, state=tk.DISABLED,
            relief=tk.FLAT, padx=6, pady=6,
        )
        self._detail_text.pack(fill=tk.BOTH, expand=False, pady=(4, 0))

    def _build_findings_list(self, parent: tk.Widget, category: str) -> None:
        cols = ("sev", "title", "nodes")
        tree = ttk.Treeview(parent, columns=cols, show="headings", selectmode="browse")
        tree.heading("sev",   text="Sev")
        tree.heading("title", text="Title")
        tree.heading("nodes", text="Nodes")
        tree.column("sev",   width=65,  stretch=False)
        tree.column("title", width=260, stretch=True)
        tree.column("nodes", width=70,  stretch=False)

        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        for sev, colour in SEV_COLOURS.items():
            tree.tag_configure(f"sev_{sev}", foreground=colour)

        tree.bind("<<TreeviewSelect>>", lambda e, t=tree: self._on_finding_selected(e, t))
        setattr(self, f"_ftree_{category}", tree)

    def _build_lifetime_panel(self) -> None:
        frame = tk.LabelFrame(self, text=" Resource Lifetimes ",
                               bg=PANEL_BG, fg=DARK_FG, font=("Segoe UI Semibold", 9),
                               bd=1, relief=tk.GROOVE)
        frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        # Canvas + both scrollbars in a grid so they share the corner
        canvas_frame = tk.Frame(frame, bg=DARK_BG)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self._lifetime_canvas = tk.Canvas(canvas_frame, bg=DARK_BG, highlightthickness=0)
        vsb = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL,   command=self._lifetime_canvas.yview)
        hsb = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self._lifetime_canvas.xview)
        self._lifetime_canvas.configure(
            xscrollcommand=hsb.set,
            yscrollcommand=vsb.set,
        )
        self._lifetime_canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Mouse-wheel vertical scroll
        self._lifetime_canvas.bind("<MouseWheel>", self._on_lifetime_scroll)

        # Cursor line — drawn on top, updated when user selects a node
        self._cursor_line: Optional[int] = None

        # Tooltip label
        self._lt_tip_var = tk.StringVar()
        self._lt_tip = tk.Label(frame, textvariable=self._lt_tip_var,
                                 bg="#ffffe0", fg="#000000", font=("Segoe UI", 8),
                                 relief=tk.SOLID, borderwidth=1, padx=4)
        # Not packed — placed via place() on hover

        self._lifetime_canvas.bind("<Motion>",  self._on_lifetime_hover)
        self._lifetime_canvas.bind("<Leave>",   self._on_lifetime_leave)

    def _build_status_bar(self) -> None:
        self._status_var = tk.StringVar(value="Ready  |  Open a trace JSON to analyze, or load a saved result")
        bar = tk.Label(self, textvariable=self._status_var, bg=ACCENT, fg="white",
                        anchor="w", padx=8, font=("Segoe UI", 8))
        bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------
    # File opening
    # ------------------------------------------------------------------

    def _auto_open(self, path: str) -> None:
        """Called on startup with a command-line path — detect type and load."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # A result JSON has a "findings" key at the top level
            if "findings" in data and "trace" in data:
                self._load_result_dict(data, source_path=path)
            else:
                # Treat as raw trace — run the engine
                self._run_analysis(path, data)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _open_trace_dialog(self) -> None:
        """Let the user pick a raw trace JSON and run the engine on it."""
        path = filedialog.askopenfilename(
            title="Open HAL Trace",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            messagebox.showerror("Parse error", str(exc))
            return
        self._run_analysis(path, data)

    def _open_result_dialog(self) -> None:
        """Let the user pick a pre-computed result JSON (skips analysis)."""
        path = filedialog.askopenfilename(
            title="Open Analysis Result JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._load_result_dict(data, source_path=path)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ------------------------------------------------------------------
    # Analysis — runs in a background thread so the UI stays responsive
    # ------------------------------------------------------------------

    def _run_analysis(self, path: str, data: Dict[str, Any]) -> None:
        arch = self._arch_var.get().strip() or data.get("arch") or None
        self._status(f"Analyzing {Path(path).name} ...")
        self._source_label.config(text=f"Analyzing {Path(path).name}...")

        def _worker():
            try:
                result = self._engine.analyze_dict(data, arch_hint=arch, source_path=path)
                # Schedule UI update back on the main thread
                self.after(0, lambda: self._load_result(result))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Analysis error", str(exc)))
                self.after(0, lambda: self._status("Analysis failed."))

        threading.Thread(target=_worker, daemon=True).start()

    def _load_result(self, result: AnalysisResult) -> None:
        """Called on the main thread after the engine finishes."""
        self._model = DisplayModel.from_result(result)
        self._populate_all()
        n = len(self._model.findings)
        crit = self._model.summary.get("critical", 0)
        high = self._model.summary.get("high", 0)
        self._source_label.config(
            text=f"{Path(self._model.source).name}  |  arch: {self._model.arch or '(none)'}"
        )
        self._status(
            f"Done  |  {len(self._model.nodes)} nodes  |  {len(self._model.resources)} resources  |  "
            f"{n} findings  ({crit} critical, {high} high)"
        )

    def _load_result_dict(self, data: Dict[str, Any], source_path: str) -> None:
        """Load and display a pre-computed result dict directly."""
        self._model = DisplayModel(data)
        self._populate_all()
        n = len(self._model.findings)
        self._source_label.config(
            text=f"{Path(source_path).name}  |  arch: {self._model.arch or '(none)'}  [saved result]"
        )
        self._status(
            f"Loaded result  |  {len(self._model.nodes)} nodes  |  "
            f"{len(self._model.resources)} resources  |  {n} findings"
        )

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def _populate_all(self) -> None:
        self._populate_call_sequence()
        self._refresh_findings_panel()
        self._populate_lifetime_timeline()

    def _populate_call_sequence(self) -> None:
        tree = self._seq_tree
        tree.delete(*tree.get_children())
        if not self._model:
            return
        for i, node in enumerate(self._model.nodes):
            idx      = node.get("index", i)
            op       = node.get("op", "?")
            cat      = node.get("category", "")
            blocking = node.get("blocking", "")
            queue    = node.get("queue") or ""
            max_sev  = self._model.max_severity_for_node(idx)

            flags = ""
            if node.get("sync_point"):
                flags += "[S] "
            if self._model.findings_for_node(idx):
                flags += "[!]"

            tags = [f"sev_{max_sev}"] if max_sev else ["even" if i % 2 == 0 else "odd"]
            tree.insert("", tk.END, iid=str(idx),
                         values=(idx, op, cat, blocking, queue, flags), tags=tags)

    def _refresh_findings_panel(self) -> None:
        if not self._model:
            return
        sev_order = ["info", "low", "medium", "high", "critical"]
        threshold = sev_order.index(self._sev_var.get())
        filtered = [
            f for f in self._model.findings
            if sev_order.index(f.get("severity", "info")) >= threshold
        ]
        for key in ("all", "resource_contention", "fragility", "performance"):
            tree: ttk.Treeview = getattr(self, f"_ftree_{key}", None)
            if tree is None:
                continue
            tree.delete(*tree.get_children())
            subset = filtered if key == "all" else [f for f in filtered if f.get("category") == key]
            for f in subset:
                sev   = f.get("severity", "info")
                title = f.get("title", "")[:60]
                nodes = str(f.get("related_nodes", []))[:20]
                tree.insert("", tk.END, values=(sev.upper(), title, nodes),
                             tags=[f"sev_{sev}"])

    # Layout constants shared between populate and cursor/hover helpers
    _LT_PX          = 14    # pixels per node slot
    _LT_LEFT_MARGIN = 130   # width of resource name label column
    _LT_ROW_H       = 18
    _LT_ROW_PAD     = 4
    _LT_AXIS_H      = 22    # height of the X-axis ruler at the bottom

    def _populate_lifetime_timeline(self) -> None:
        canvas = self._lifetime_canvas
        canvas.delete("all")
        self._cursor_line = None

        if not self._model or not self._model.lifetimes:
            canvas.create_text(10, 20, anchor="w", text="No resource lifetime data.",
                                fill="#666666", font=("Segoe UI", 9))
            return

        PX   = self._LT_PX
        LM   = self._LT_LEFT_MARGIN
        RH   = self._LT_ROW_H
        RP   = self._LT_ROW_PAD
        AH   = self._LT_AXIS_H
        n    = max(len(self._model.nodes), 1)

        # ── Legend strip (pinned at top of canvas) ─────────────────────
        LEGEND_H = 20
        legend_items = [
            ("buffer",    "Buffer"),
            ("texture",   "Texture"),
            ("fence",     "Fence"),
            ("semaphore", "Semaphore"),
            ("queue",     "Queue"),
            ("pipeline",  "Pipeline"),
            ("descriptor","Descriptor"),
            ("unknown",   "Other"),
        ]
        lx = 4
        canvas.create_text(lx, LEGEND_H // 2, anchor="w",
                             text="KIND:", fill="#888888", font=("Segoe UI", 7, "bold"))
        lx += 38
        for kind, label in legend_items:
            col = RES_KIND_COLOURS.get(kind, "#888888")
            canvas.create_rectangle(lx, 5, lx + 10, 15, fill=col, outline="")
            lx += 14
            canvas.create_text(lx, LEGEND_H // 2, anchor="w",
                                 text=label, fill="#aaaaaa", font=("Segoe UI", 7))
            lx += len(label) * 5 + 10

        # Mapped / never-freed legend entries
        canvas.create_rectangle(lx, 5, lx + 10, 15, fill="#ffffff",
                                  outline="", stipple="gray50")
        lx += 14
        canvas.create_text(lx, LEGEND_H // 2, anchor="w",
                             text="CPU mapped", fill="#aaaaaa", font=("Segoe UI", 7))
        lx += 74
        canvas.create_rectangle(lx, 5, lx + 10, 15, fill="#FF4444", outline="")
        lx += 14
        canvas.create_text(lx, LEGEND_H // 2, anchor="w",
                             text="Never freed", fill="#aaaaaa", font=("Segoe UI", 7))

        # Separator line under legend
        canvas.create_line(0, LEGEND_H, 9999, LEGEND_H, fill="#333333")

        TOP_OFFSET = LEGEND_H + 4   # rows start below the legend

        # ── Rows: one per resource ─────────────────────────────────────
        # Store row metadata for hover hit-testing
        self._lt_rows: list = []   # list of (y0, y1, name, lt_dict, size_b)

        row = 0
        for name, lt in self._model.lifetimes.items():
            res    = self._model.resources.get(name, {})
            kind   = res.get("kind", "unknown")
            colour = RES_KIND_COLOURS.get(kind, "#888888")
            size_b = res.get("size_bytes", 0)
            y0     = TOP_OFFSET + row * (RH + RP)
            y1     = y0 + RH

            # Alternating row background for readability
            row_bg = "#242424" if row % 2 == 0 else "#1e1e1e"
            canvas.create_rectangle(0, y0, 9999, y1, fill=row_bg, outline="")

            # Kind chip (colour swatch) in left margin
            canvas.create_rectangle(4, y0 + 3, 14, y1 - 3,
                                      fill=colour, outline="")

            # Resource name label (truncated, left-aligned after swatch)
            canvas.create_text(18, (y0 + y1) / 2, anchor="w",
                                 text=name[:20], fill=DARK_FG, font=("Consolas", 8))

            # Live-range bar
            alloc_x = LM + lt.get("alloc_index", 0) * PX
            free_i  = lt.get("free_index")
            free_x  = LM + (free_i if free_i is not None else n) * PX
            canvas.create_rectangle(alloc_x, y0 + 2, free_x, y1 - 2,
                                      fill=colour, outline="#444444")

            # "Never freed" marker — red right-edge stripe
            if free_i is None:
                canvas.create_rectangle(free_x - 4, y0 + 2, free_x, y1 - 2,
                                          fill="#FF4444", outline="")

            # CPU-mapped intervals (white stipple overlay)
            for m_start, m_end in lt.get("map_intervals", []):
                mx0 = LM + m_start * PX
                mx1 = LM + (m_end if m_end is not None else n) * PX
                canvas.create_rectangle(mx0, y0 + 4, mx1, y1 - 4,
                                          fill="#ffffff", outline="", stipple="gray50")

            self._lt_rows.append((y0, y1, name, lt, size_b))
            row += 1

        # ── X-axis ruler ───────────────────────────────────────────────
        rows_h  = TOP_OFFSET + row * (RH + RP)
        axis_y  = rows_h + 2
        total_w = LM + n * PX + 20
        total_h = axis_y + AH

        canvas.create_line(LM, axis_y, LM + n * PX, axis_y, fill="#555555")

        # Adaptive tick spacing: aim for ~20 ticks
        step = max(1, n // 20)
        step = 5 * max(1, (step + 4) // 5)  # round up to nearest multiple of 5
        for i in range(0, n + 1, step):
            x = LM + i * PX
            canvas.create_line(x, axis_y, x, axis_y + 4, fill="#777777")
            canvas.create_text(x, axis_y + 6, anchor="n", text=str(i),
                                 fill="#888888", font=("Consolas", 7))

        # Axis label
        canvas.create_text(LM + (n * PX) // 2, axis_y + 14, anchor="n",
                             text="Node index  (position in call sequence)",
                             fill="#666666", font=("Segoe UI", 7))

        # Divider between label column and bar area
        canvas.create_line(LM, TOP_OFFSET, LM, axis_y, fill="#333333")

        # Left-column header
        canvas.create_text(LM // 2, TOP_OFFSET - 2, anchor="s",
                             text="Resource", fill="#666666", font=("Segoe UI", 7))

        canvas.configure(scrollregion=(0, 0, total_w, total_h))

    def _lifetime_node_at_x(self, canvas_x: int) -> Optional[int]:
        """Return the node index corresponding to a canvas X coordinate."""
        lm = self._LT_LEFT_MARGIN
        if canvas_x < lm:
            return None
        idx = (canvas_x - lm) // self._LT_PX
        if not self._model:
            return None
        return min(idx, len(self._model.nodes) - 1)

    def _on_lifetime_hover(self, event: tk.Event) -> None:
        """Show a tooltip describing the resource and node under the cursor."""
        if not self._model:
            return
        cx = self._lifetime_canvas.canvasx(event.x)
        cy = self._lifetime_canvas.canvasy(event.y)

        node_idx = self._lifetime_node_at_x(int(cx))

        hit_name: Optional[str] = None
        hit_lt = None
        hit_size = 0
        for (y0, y1, name, lt, size_b) in getattr(self, "_lt_rows", []):
            if y0 <= cy <= y1:
                hit_name = name
                hit_lt   = lt
                hit_size = size_b
                break

        if hit_name is None and node_idx is None:
            self._lt_tip.place_forget()
            return

        parts = []
        if hit_name:
            res  = self._model.resources.get(hit_name, {})
            kind = res.get("kind", "unknown")
            mem  = res.get("memory", "")
            ai   = hit_lt.get("alloc_index", "?")
            fi   = hit_lt.get("free_index")
            maps = len(hit_lt.get("map_intervals", []))
            size_s = f"{hit_size // 1024} KiB" if hit_size >= 1024 else (f"{hit_size} B" if hit_size else "")
            fi_s = str(fi) if fi is not None else "never freed"
            parts.append(f"{hit_name}  [{kind}]  {mem}  {size_s}")
            parts.append(f"live: node {ai} -> {fi_s}  |  mapped {maps}x")

        if node_idx is not None and self._model.nodes:
            nodes = self._model.nodes
            if 0 <= node_idx < len(nodes):
                nd = nodes[node_idx]
                op = nd.get("op", "?")
                parts.append(f"node {node_idx}: {op}")

        if parts:
            self._lt_tip_var.set("  |  ".join(parts))
            self._lt_tip.place(x=event.x + 12, y=event.y - 20)
        else:
            self._lt_tip.place_forget()

    def _on_lifetime_scroll(self, event: tk.Event) -> None:
        """Scroll the lifetime canvas vertically with the mouse wheel."""
        self._lifetime_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_lifetime_leave(self, _event: tk.Event) -> None:
        self._lt_tip.place_forget()

    def _draw_lifetime_cursor(self, node_idx: int) -> None:
        """Draw a vertical cursor line on the timeline at the given node index."""
        canvas = self._lifetime_canvas
        if self._cursor_line is not None:
            canvas.delete(self._cursor_line)
            self._cursor_line = None
        if not self._model or not self._model.lifetimes:
            return
        x = self._LT_LEFT_MARGIN + node_idx * self._LT_PX
        LEGEND_H  = 20
        TOP_OFFSET = LEGEND_H + 4
        n_rows = len(self._lt_rows) if hasattr(self, "_lt_rows") else 1
        total_h = TOP_OFFSET + n_rows * (self._LT_ROW_H + self._LT_ROW_PAD) + self._LT_AXIS_H + 10
        self._cursor_line = canvas.create_line(
            x, TOP_OFFSET, x, total_h, fill="#ffffff", dash=(3, 3), width=1
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_node_selected(self, event: tk.Event) -> None:
        sel = self._seq_tree.selection()
        if not sel or not self._model:
            return
        node_idx = int(sel[0])
        # Sync cursor on lifetime timeline
        self._draw_lifetime_cursor(node_idx)
        findings = self._model.findings_for_node(node_idx)
        if findings:
            self._show_finding_detail(findings[0])
        else:
            node = next((n for n in self._model.nodes if n.get("index") == node_idx), {})
            self._show_node_detail(node)

    def _on_finding_selected(self, event: tk.Event, tree: ttk.Treeview) -> None:
        sel = tree.selection()
        if not sel or not self._model:
            return
        title = tree.item(sel[0])["values"][1]
        match = next(
            (f for f in self._model.findings if f.get("title", "").startswith(title[:40])),
            None,
        )
        if match:
            self._show_finding_detail(match)
            nodes = match.get("related_nodes", [])
            if nodes:
                self._seq_tree.see(str(nodes[0]))
                self._seq_tree.selection_set(str(nodes[0]))

    def _show_finding_detail(self, finding: Dict[str, Any]) -> None:
        txt = self._detail_text
        txt.configure(state=tk.NORMAL)
        txt.delete("1.0", tk.END)
        sev   = finding.get("severity", "info").upper()
        cat   = finding.get("category", "")
        title = finding.get("title", "")
        expl  = finding.get("explanation", "")
        sugg  = finding.get("suggestion", "")
        nodes = finding.get("related_nodes", [])
        content = (
            f"[{sev}]  {title}\n"
            f"Category: {cat}   Nodes: {nodes}\n"
            f"{'-'*60}\n"
            f"{expl}\n"
        )
        if sugg:
            content += f"\n-> Suggestion:\n{sugg}\n"
        txt.insert(tk.END, content)
        txt.configure(state=tk.DISABLED)

    def _show_node_detail(self, node: Dict[str, Any]) -> None:
        txt = self._detail_text
        txt.configure(state=tk.NORMAL)
        txt.delete("1.0", tk.END)
        content = (
            f"Node [{node.get('index','?')}]  {node.get('op','')}\n"
            f"Category : {node.get('category','')}\n"
            f"Blocking : {node.get('blocking','')}\n"
            f"Queue    : {node.get('queue') or '-'}\n"
            f"Reads    : {node.get('resources_read', [])}\n"
            f"Writes   : {node.get('resources_write', [])}\n"
        )
        txt.insert(tk.END, content)
        txt.configure(state=tk.DISABLED)

    def _status(self, msg: str) -> None:
        self._status_var.set(msg)


# ──────────────────────────────────────────────────────────────────────────────
# Tiny tooltip helper
# ──────────────────────────────────────────────────────────────────────────────

def _tooltip(widget: tk.Widget, text: str) -> None:
    tip: Optional[tk.Toplevel] = None

    def show(event: tk.Event) -> None:
        nonlocal tip
        tip = tk.Toplevel(widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{event.x_root+12}+{event.y_root+6}")
        tk.Label(tip, text=text, background="#ffffe0", foreground="#000000",
                  relief=tk.SOLID, borderwidth=1, font=("Segoe UI", 8)).pack()

    def hide(_event: tk.Event) -> None:
        nonlocal tip
        if tip:
            tip.destroy()
            tip = None

    widget.bind("<Enter>", show)
    widget.bind("<Leave>", hide)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    app = HALAnalyzerApp(initial_file=initial)
    app.mainloop()


if __name__ == "__main__":
    main()
