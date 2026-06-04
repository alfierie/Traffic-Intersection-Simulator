"""
============================================================
 STOCHASTIC TRAFFIC SIMULATION — 4-Way Intersection
 with Emergency Vehicle (Ambulance) Priority
 + Real-time Animated Playback
============================================================
 Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque
import pandas as pd
import copy

# ──────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Intersection Simulation",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.main { background: #0d0d0f; }
.stApp { background: #0d0d0f; color: #e8e8e8; }
h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

.title-block {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #e94560;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
}
.title-block h1 { color: #e94560; font-size: 2.2rem; margin: 0; }
.title-block p  { color: #a0aec0; margin: 6px 0 0 0; font-family: 'Space Mono', monospace; font-size: 0.85rem; }

.metric-card {
    background: #1a1a2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-card .val { font-size: 2rem; font-weight: 800; color: #63b3ed; font-family: 'Space Mono', monospace; }
.metric-card .lbl { font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: 1px; }

.alert-red {
    background: #2d1515; border-left: 4px solid #e94560;
    padding: 10px 16px; border-radius: 6px; color: #fc8181;
    font-family: 'Space Mono', monospace; font-size: 0.82rem;
}
.alert-green {
    background: #143020; border-left: 4px solid #48bb78;
    padding: 10px 16px; border-radius: 6px; color: #9ae6b4;
    font-family: 'Space Mono', monospace; font-size: 0.82rem;
}
.section-header {
    border-bottom: 2px solid #e94560; padding-bottom: 6px;
    margin-top: 28px; margin-bottom: 16px; color: #e8e8e8;
    font-size: 1.1rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 2px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────
DIRECTIONS = ["North", "South", "East", "West"]

# Intersection geometry (in data units 0-10)
ROAD_LO, ROAD_HI = 3.5, 6.5   # road band width
MID = 5.0

# Entry points for each lane (where vehicles queue outside intersection)
ENTRY = {
    "North": (4.6, 3.2),   # waiting just south of box, moving north
    "South": (5.4, 6.8),   # waiting just north of box, moving south
    "East":  (3.2, 5.4),   # waiting just right of box, moving east  -- FIXED: was 4.6
    "West":  (6.8, 4.6),   # waiting just left  of box, moving west  -- FIXED: was 5.4
}

# Exit points (where vehicles disappear)
EXIT = {
    "North": (4.6, 9.8),
    "South": (5.4, 0.2),
    "East":  (0.2, 5.4),
    "West":  (9.8, 4.6),
}

# Queue stacking direction (away from intersection)
QUEUE_DELTA = {
    "North": (0,  -0.62),
    "South": (0,  +0.62),
    "East":  (-0.62, 0),
    "West":  (+0.62, 0),
}

TL_POSITIONS = {
    "North": (3.1, 6.7),
    "South": (6.9, 3.3),
    "East":  (3.3, 3.1),
    "West":  (6.7, 6.9),
}

# ──────────────────────────────────────────────
#  DATACLASSES
# ──────────────────────────────────────────────
@dataclass
class Vehicle:
    vid: int
    direction: str
    arrival_time: float
    vehicle_type: str       # "car" | "ambulance"
    wait_time: float = 0.0
    served: bool = False
    departure_time: float = 0.0
    # Animation state
    x: float = 0.0
    y: float = 0.0
    crossing: bool = False  # True while moving through intersection
    cross_progress: float = 0.0  # 0→1

    def __post_init__(self):
        ex, ey = ENTRY[self.direction]
        self.x, self.y = ex, ey


@dataclass
class SimState:
    t: float = 0.0
    step: int = 0
    queues: Dict[str, deque] = field(default_factory=lambda: {d: deque() for d in DIRECTIONS})
    crossing_vehicles: List[Vehicle] = field(default_factory=list)
    phase: int = 0
    phase_timer: float = 0.0
    emergency_active: bool = False
    emergency_dir: str = ""
    emergency_timer: float = 0.0
    next_vid: int = 0
    served_vehicles: List[Vehicle] = field(default_factory=list)
    total_arrivals: int = 0
    ambulance_arrivals: int = 0
    ambulance_wait_times: List[float] = field(default_factory=list)
    car_wait_times: List[float] = field(default_factory=list)
    queue_history: List[Dict] = field(default_factory=list)
    phase_history: List[Tuple[float, int]] = field(default_factory=list)


# ──────────────────────────────────────────────
#  SIMULATION ENGINE  (stores per-frame snapshots)
# ──────────────────────────────────────────────
class TrafficSimulation:
    def __init__(self, params: dict):
        self.p = params
        self.rng = np.random.default_rng(params.get("seed", 42))
        self.state = SimState()

    def _interarrival(self, direction: str) -> float:
        rate = self.p["arrival_rates"][direction]
        return self.rng.exponential(1.0 / rate) if rate > 0 else 9999.0

    def _service_time(self) -> float:
        return self.rng.exponential(self.p["mean_service_time"])

    def _snapshot(self, s: SimState) -> dict:
        """Lightweight frame snapshot for animation playback."""
        return {
            "t": s.t,
            "phase": s.phase,
            "emergency_active": s.emergency_active,
            "emergency_dir": s.emergency_dir,
            "queues": {d: [(v.vid, v.vehicle_type, v.x, v.y) for v in s.queues[d]]
                       for d in DIRECTIONS},
            "crossing": [(v.vid, v.vehicle_type, v.x, v.y) for v in s.crossing_vehicles],
            "total_arrivals": s.total_arrivals,
            "served": len(s.served_vehicles),
        }

    def run(self, duration: float, record_every: float = 0.5) -> Tuple[SimState, List[dict]]:
        s = self.state
        next_arr = {d: s.t + self._interarrival(d) for d in DIRECTIONS}
        green_ns  = self.p["green_ns"]
        green_ew  = self.p["green_ew"]
        amb_prob  = self.p["ambulance_prob"]
        emg_dur   = self.p["emergency_duration"]
        dt        = self.p["dt"]
        cross_spd = self.p.get("cross_speed", 0.08)  # progress units per dt

        frames: List[dict] = []
        last_record = -record_every

        while s.t < duration:
            # ── Arrivals ──────────────────────────────────
            for d in DIRECTIONS:
                while next_arr[d] <= s.t:
                    vtype = "ambulance" if self.rng.random() < amb_prob else "car"
                    v = Vehicle(vid=s.next_vid, direction=d,
                                arrival_time=s.t, vehicle_type=vtype)
                    # Stack in queue visually
                    idx = len(s.queues[d])
                    dx, dy = QUEUE_DELTA[d]
                    ex, ey = ENTRY[d]
                    v.x = ex + idx * dx
                    v.y = ey + idx * dy
                    s.queues[d].append(v)
                    s.next_vid      += 1
                    s.total_arrivals += 1
                    if vtype == "ambulance":
                        s.ambulance_arrivals += 1
                    next_arr[d] += self._interarrival(d)

            # ── Emergency detection ───────────────────────
            if not s.emergency_active:
                for d in DIRECTIONS:
                    for v in s.queues[d]:
                        if v.vehicle_type == "ambulance":
                            s.emergency_active = True
                            s.emergency_dir    = d
                            s.emergency_timer  = 0.0
                            break
                    if s.emergency_active:
                        break

            # ── Signal phase logic ────────────────────────
            if s.emergency_active:
                s.emergency_timer += dt
                s.phase = 0 if s.emergency_dir in ("North", "South") else 1
                if s.emergency_timer >= emg_dur:
                    s.emergency_active = False
                    s.emergency_dir    = ""
                    s.emergency_timer  = 0.0
                    s.phase_timer      = 0.0
            else:
                s.phase_timer += dt
                cycle_len = green_ns if s.phase == 0 else green_ew
                if s.phase_timer >= cycle_len:
                    s.phase       = 1 - s.phase
                    s.phase_timer = 0.0

            # ── Advance crossing vehicles ─────────────────
            still_crossing = []
            for v in s.crossing_vehicles:
                v.cross_progress += cross_spd
                t_prog = min(v.cross_progress, 1.0)
                ex, ey = ENTRY[v.direction]
                fx, fy = EXIT[v.direction]
                v.x = ex + (fx - ex) * t_prog
                v.y = ey + (fy - ey) * t_prog
                if v.cross_progress >= 1.0:
                    v.served         = True
                    v.departure_time = s.t
                    s.served_vehicles.append(v)
                    if v.vehicle_type == "ambulance":
                        s.ambulance_wait_times.append(v.wait_time)
                    else:
                        s.car_wait_times.append(v.wait_time)
                else:
                    still_crossing.append(v)
            s.crossing_vehicles = still_crossing

            # ── Dispatch head-of-queue on green ──────────
            active_dirs = ["North", "South"] if s.phase == 0 else ["East", "West"]
            for d in active_dirs:
                if s.queues[d] and len([v for v in s.crossing_vehicles
                                        if v.direction == d]) == 0:
                    v = s.queues[d].popleft()
                    v.crossing       = True
                    v.cross_progress = 0.0
                    ex, ey = ENTRY[d]
                    v.x, v.y = ex, ey
                    s.crossing_vehicles.append(v)
                    # Re-stack remaining queue
                    for i, qv in enumerate(s.queues[d]):
                        dx, dy = QUEUE_DELTA[d]
                        qv.x = ex + i * dx
                        qv.y = ey + i * dy

            # Accumulate wait for queued vehicles on red
            red_dirs = ["East", "West"] if s.phase == 0 else ["North", "South"]
            for d in red_dirs:
                for v in s.queues[d]:
                    v.wait_time += dt
            for d in active_dirs:
                for v in s.queues[d]:
                    v.wait_time += dt

            # ── Record frame ──────────────────────────────
            if s.t - last_record >= record_every:
                frames.append(self._snapshot(s))
                s.queue_history.append({
                    "t":      s.t,
                    "North":  len(s.queues["North"]),
                    "South":  len(s.queues["South"]),
                    "East":   len(s.queues["East"]),
                    "West":   len(s.queues["West"]),
                    "phase":  s.phase,
                    "emergency": int(s.emergency_active),
                })
                last_record = s.t

            s.t    += dt
            s.step += 1

        return s, frames


# ──────────────────────────────────────────────
#  DRAW ONE ANIMATION FRAME
# ──────────────────────────────────────────────
def draw_frame(frame: dict, figsize=(6, 6)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d0d0f")
    ax.set_facecolor("#0d0d0f")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    phase             = frame["phase"]
    emergency_active  = frame["emergency_active"]
    emergency_dir     = frame["emergency_dir"]

    # ── Road surface ──────────────────────────────
    ax.fill_between([ROAD_LO, ROAD_HI], 0, 10, color="#1c1c1e", zorder=1)
    ax.fill_betweenx([ROAD_LO, ROAD_HI], 0, 10, color="#1c1c1e", zorder=1)
    ax.add_patch(mpatches.Rectangle((ROAD_LO, ROAD_LO), 3, 3, color="#242424", zorder=2))

    # Pavement texture lines
    for v in np.arange(0.5, 10, 1.0):
        ax.plot([ROAD_LO, ROAD_HI], [v, v], color="#2a2a2a", lw=0.4, zorder=2)
        ax.plot([v, v], [ROAD_LO, ROAD_HI], color="#2a2a2a", lw=0.4, zorder=2)

    # Centre dashes
    dash_kw = dict(color="#ccaa00", lw=1.0, dashes=(4, 4), zorder=3)
    ax.plot([MID, MID], [0, ROAD_LO],    **dash_kw)
    ax.plot([MID, MID], [ROAD_HI, 10],   **dash_kw)
    ax.plot([0, ROAD_LO],   [MID, MID],  **dash_kw)
    ax.plot([ROAD_HI, 10],  [MID, MID],  **dash_kw)

    # Zebra crossings
    for i in range(4):
        off = 0.18 + i * 0.22
        ax.add_patch(mpatches.Rectangle((ROAD_LO + 0.1, ROAD_HI + off), 2.8, 0.14,
                                         color="#383838", zorder=3))
        ax.add_patch(mpatches.Rectangle((ROAD_LO + 0.1, ROAD_LO - off - 0.14), 2.8, 0.14,
                                         color="#383838", zorder=3))
        ax.add_patch(mpatches.Rectangle((ROAD_HI + off, ROAD_LO + 0.1), 0.14, 2.8,
                                         color="#383838", zorder=3))
        ax.add_patch(mpatches.Rectangle((ROAD_LO - off - 0.14, ROAD_LO + 0.1), 0.14, 2.8,
                                         color="#383838", zorder=3))

    # ── Traffic lights ────────────────────────────
    ns_col = "#48bb78" if phase == 0 else "#e94560"
    ew_col = "#48bb78" if phase == 1 else "#e94560"
    if emergency_active:
        ns_col = "#48bb78" if emergency_dir in ("North", "South") else "#e94560"
        ew_col = "#48bb78" if emergency_dir in ("East", "West")   else "#e94560"

    tl_map = {
        "North": ns_col, "South": ns_col,
        "East":  ew_col, "West":  ew_col,
    }
    for d, (tx, ty) in TL_POSITIONS.items():
        col = tl_map[d]
        # Housing
        ax.add_patch(mpatches.FancyBboxPatch(
            (tx - 0.22, ty - 0.22), 0.44, 0.44,
            boxstyle="round,pad=0.04", facecolor="#111", edgecolor="#444", lw=1, zorder=7))
        # Glow
        if col == "#48bb78":
            ax.add_patch(mpatches.Circle((tx, ty), 0.28, color=col, alpha=0.18, zorder=6))
        ax.add_patch(mpatches.Circle((tx, ty), 0.16, color=col, zorder=8))

    # ── Direction labels ──────────────────────────
    for txt, x, y in [("N", 5.0, 9.65), ("S", 5.0, 0.3),
                       ("E", 0.3, 5.0),  ("W", 9.7, 5.0)]:
        ax.text(x, y, txt, ha="center", va="center", fontsize=8,
                color="#555", fontweight="bold", fontfamily="monospace", zorder=4)

    # ── Vehicle drawing helper ────────────────────
    def draw_vehicle(x, y, vtype, direction, alpha=1.0):
        is_amb = vtype == "ambulance"
        body_col  = "#f6e05e" if is_amb else "#63b3ed"
        roof_col  = "#c8920a" if is_amb else "#2b6cb0"

        # Orientation
        if direction in ("North", "South"):
            w, h = 0.30, 0.48
        else:
            w, h = 0.48, 0.30

        # Shadow
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - w + 0.06, y - h + 0.06), w * 2, h * 2,
            boxstyle="round,pad=0.05",
            facecolor="#000", alpha=0.3 * alpha, zorder=4))
        # Body
        body = mpatches.FancyBboxPatch(
            (x - w, y - h), w * 2, h * 2,
            boxstyle="round,pad=0.06",
            facecolor=body_col, edgecolor="#000", lw=0.7, zorder=5, alpha=alpha)
        ax.add_patch(body)
        # Roof stripe
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - w * 0.55, y - h * 0.55), w * 1.1, h * 1.1,
            boxstyle="round,pad=0.03",
            facecolor=roof_col, alpha=0.55 * alpha, zorder=6))

        if is_amb:
            # Red cross
            ax.plot([x - 0.10, x + 0.10], [y, y],
                    color="#e94560", lw=1.5, zorder=7, alpha=alpha)
            ax.plot([x, x], [y - 0.10, y + 0.10],
                    color="#e94560", lw=1.5, zorder=7, alpha=alpha)
            # Flashing light bar (always on for visibility)
            ax.add_patch(mpatches.Rectangle(
                (x - w * 0.45, y + h * 0.5), w * 0.9, h * 0.28,
                facecolor="#e94560", alpha=0.9 * alpha, zorder=8))

    # Draw queued vehicles
    for d in DIRECTIONS:
        for _, vtype, vx, vy in frame["queues"][d]:
            draw_vehicle(vx, vy, vtype, d, alpha=0.85)

    # Draw crossing vehicles (full opacity, slightly larger)
    for _, vtype, vx, vy in frame["crossing"]:
        d = None  # derive direction from position
        # Infer direction: crossing vehicles move along their original lane
        if abs(vx - 4.6) < 0.4:
            d = "North"
        elif abs(vx - 5.4) < 0.4:
            d = "South"
        elif abs(vy - 5.4) < 0.4:
            d = "East"
        else:
            d = "West"
        draw_vehicle(vx, vy, vtype, d, alpha=1.0)

    # ── Emergency overlay ─────────────────────────
    if emergency_active:
        ax.text(MID, MID, "EMERGENCY\nPREEMPTION",
                ha="center", va="center", fontsize=8.5,
                color="#f6e05e", fontweight="bold", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a0000",
                          edgecolor="#e94560", lw=1.5, alpha=0.92),
                zorder=12)

    # ── HUD ───────────────────────────────────────
    phase_txt  = "NS GREEN / EW RED" if phase == 0 else "EW GREEN / NS RED"
    phase_col  = "#48bb78"
    t_str      = f"T = {frame['t']:.0f}s"
    ax.text(0.5, 9.75, t_str,     ha="left",   va="top", fontsize=7.5,
            color="#718096", fontfamily="monospace", zorder=10)
    ax.text(9.5, 9.75, phase_txt, ha="right",  va="top", fontsize=7.0,
            color=phase_col,  fontfamily="monospace", fontweight="bold", zorder=10)

    # Queue count badges
    badge_pos = {
        "North": (4.6, 0.55), "South": (5.4, 9.45),
        "East":  (0.55, 5.4), "West":  (9.45, 4.6),
    }
    for d in DIRECTIONS:
        qlen = len(frame["queues"][d])
        if qlen > 0:
            bx, by = badge_pos[d]
            ax.add_patch(mpatches.Circle((bx, by), 0.35, color="#e94560", zorder=9))
            ax.text(bx, by, str(qlen), ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold",
                    fontfamily="monospace", zorder=10)

    plt.tight_layout(pad=0.1)
    return fig


# ──────────────────────────────────────────────
#  STATIC CHARTS
# ──────────────────────────────────────────────
def plot_queue_history(history: List[Dict]):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), facecolor="#0d0d0f")

    colors = {"North": "#63b3ed", "South": "#48bb78", "East": "#f6ad55", "West": "#fc8181"}
    ax1 = axes[0]
    ax1.set_facecolor("#0d0d0f")
    for d, c in colors.items():
        ax1.plot(df["t"], df[d], color=c, lw=1.4, label=d)
    emg_mask = df["emergency"] == 1
    if emg_mask.any():
        ymax = df[["North","South","East","West"]].values.max()
        ax1.fill_between(df["t"], 0, ymax, where=emg_mask,
                         color="#e94560", alpha=0.12, label="Emergency")
    ax1.set_ylabel("Queue Length", color="#a0aec0", fontsize=9)
    ax1.tick_params(colors="#718096", labelsize=8)
    ax1.spines[:].set_color("#2d3748")
    ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="#e8e8e8",
               framealpha=0.9, loc="upper left")
    ax1.set_title("Queue Lengths Over Time", color="#e8e8e8", fontsize=10, pad=6)

    ax2 = axes[1]
    ax2.set_facecolor("#0d0d0f")
    phase_colors = {0: "#48bb78", 1: "#e94560"}
    prev_t, prev_p = df["t"].iloc[0], int(df["phase"].iloc[0])
    for _, row in df.iterrows():
        p = int(row["phase"])
        if p != prev_p:
            ax2.barh(0, row["t"] - prev_t, left=prev_t, height=0.8,
                     color=phase_colors[prev_p], alpha=0.75)
            prev_t, prev_p = row["t"], p
    ax2.barh(0, df["t"].iloc[-1] - prev_t, left=prev_t, height=0.8,
             color=phase_colors[prev_p], alpha=0.75)
    ax2.set_xlim(df["t"].min(), df["t"].max())
    ax2.set_yticks([0])
    ax2.set_yticklabels(["Phase"], color="#a0aec0", fontsize=8)
    ax2.tick_params(colors="#718096", labelsize=8)
    ax2.spines[:].set_color("#2d3748")
    ax2.set_xlabel("Simulation Time (s)", color="#a0aec0", fontsize=9)
    ax2.set_title("Signal Phase Timeline  (Green = NS active,  Red = EW active)",
                  color="#e8e8e8", fontsize=9, pad=4)
    ns_p = mpatches.Patch(color="#48bb78", alpha=0.8, label="NS Green")
    ew_p = mpatches.Patch(color="#e94560", alpha=0.8, label="EW Green")
    ax2.legend(handles=[ns_p, ew_p], fontsize=8, facecolor="#1a1a2e",
               labelcolor="#e8e8e8", framealpha=0.9)

    plt.tight_layout(pad=1.2)
    return fig


def plot_wait_distribution(car_waits, amb_waits):
    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#0d0d0f")
    ax.set_facecolor("#0d0d0f")
    if car_waits:
        ax.hist(car_waits, bins=30, color="#63b3ed", alpha=0.75, label=f"Cars (n={len(car_waits)})")
    if amb_waits:
        ax.hist(amb_waits, bins=max(5, len(amb_waits)//2), color="#f6e05e",
                alpha=0.85, label=f"Ambulances (n={len(amb_waits)})")
    ax.set_xlabel("Wait Time (s)", color="#a0aec0", fontsize=9)
    ax.set_ylabel("Frequency",    color="#a0aec0", fontsize=9)
    ax.set_title("Wait Time Distribution", color="#e8e8e8", fontsize=10, pad=6)
    ax.tick_params(colors="#718096", labelsize=8)
    ax.spines[:].set_color("#2d3748")
    ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="#e8e8e8", framealpha=0.9)
    plt.tight_layout(pad=1.0)
    return fig


def plot_sensitivity(base_params: dict):
    probs = np.linspace(0.0, 0.25, 16)
    amb_means, car_means = [], []
    for p in probs:
        params = {**base_params, "ambulance_prob": float(p), "seed": 7}
        sim = TrafficSimulation(params)
        s, _ = sim.run(base_params["duration"])
        amb_means.append(np.mean(s.ambulance_wait_times) if s.ambulance_wait_times else 0)
        car_means.append(np.mean(s.car_wait_times)       if s.car_wait_times       else 0)

    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#0d0d0f")
    ax.set_facecolor("#0d0d0f")
    ax.plot(probs * 100, amb_means, color="#f6e05e", lw=2, marker="o", ms=4, label="Ambulance avg wait")
    ax.plot(probs * 100, car_means, color="#63b3ed", lw=2, marker="s", ms=4, label="Car avg wait")
    ax.set_xlabel("Ambulance Arrival Probability (%)", color="#a0aec0", fontsize=9)
    ax.set_ylabel("Mean Wait Time (s)",                color="#a0aec0", fontsize=9)
    ax.set_title("Sensitivity: Ambulance Probability vs. Mean Wait Time",
                 color="#e8e8e8", fontsize=10, pad=6)
    ax.tick_params(colors="#718096", labelsize=8)
    ax.spines[:].set_color("#2d3748")
    ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="#e8e8e8", framealpha=0.9)
    plt.tight_layout(pad=1.0)
    return fig


# ──────────────────────────────────────────────
#  MAIN APP
# ──────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="title-block">
        <h1>Traffic Intersection Simulator</h1>
        <p>Stochastic Model · 4-Way Intersection · Emergency Vehicle Preemption · Animated Playback</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────
    with st.sidebar:
        st.markdown("## Parameters")
        st.markdown("---")

        st.markdown("### Arrival Rates (veh/s)")
        r_n = st.slider("North", 0.05, 0.8, 0.25, 0.05)
        r_s = st.slider("South", 0.05, 0.8, 0.25, 0.05)
        r_e = st.slider("East",  0.05, 0.8, 0.20, 0.05)
        r_w = st.slider("West",  0.05, 0.8, 0.20, 0.05)

        st.markdown("### Signal Timing (s)")
        green_ns = st.slider("Green NS phase", 10, 90, 30, 5)
        green_ew = st.slider("Green EW phase", 10, 90, 30, 5)

        st.markdown("### Emergency")
        amb_prob = st.slider("Ambulance probability", 0.0, 0.3, 0.05, 0.01,
                              help="Probability each arriving vehicle is an ambulance")
        emg_dur  = st.slider("Preemption duration (s)", 5, 60, 20, 5,
                              help="How long the signal holds green for the emergency direction")

        st.markdown("### Simulation")
        duration  = st.slider("Duration (s)",            120, 1800, 600,  60)
        mean_svc  = st.slider("Mean service time (s/veh)", 1.0,  8.0, 3.0, 0.5)
        anim_spd  = st.slider("Animation speed (fps)",      2,    30,  10,   1)
        seed      = st.number_input("Random seed", 0, 9999, 42, 1)

        run_btn  = st.button("Run Simulation", use_container_width=True, type="primary")
        sens_btn = st.button("Run Sensitivity Analysis", use_container_width=True)

    params = {
        "arrival_rates":    {"North": r_n, "South": r_s, "East": r_e, "West": r_w},
        "green_ns":         green_ns,
        "green_ew":         green_ew,
        "ambulance_prob":   amb_prob,
        "emergency_duration": emg_dur,
        "duration":         duration,
        "mean_service_time": mean_svc,
        "dt":               0.5,
        "cross_speed":      0.10,
        "seed":             seed,
    }

    # ── Run simulation ────────────────────────────
    if run_btn or "sim_state" not in st.session_state:
        with st.spinner("Running simulation and recording frames..."):
            sim = TrafficSimulation(params)
            s, frames = sim.run(duration, record_every=0.5)
            st.session_state["sim_state"]  = s
            st.session_state["sim_frames"] = frames
            st.session_state["sim_params"] = params

    s      = st.session_state.get("sim_state")
    frames = st.session_state.get("sim_frames", [])
    if s is None:
        st.info("Configure parameters and press **Run Simulation**.")
        return

    # ── Key Metrics ───────────────────────────────
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
    total_served = len(s.served_vehicles)
    avg_car_wait = np.mean(s.car_wait_times)          if s.car_wait_times          else 0
    avg_amb_wait = np.mean(s.ambulance_wait_times)    if s.ambulance_wait_times    else 0
    total_queue  = sum(len(q) for q in s.queues.values())

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in [
        (c1, str(s.total_arrivals),    "Total Arrivals"),
        (c2, str(total_served),         "Vehicles Served"),
        (c3, f"{avg_car_wait:.1f}s",    "Avg Car Wait"),
        (c4, f"{avg_amb_wait:.1f}s",    "Avg Amb Wait"),
        (c5, str(total_queue),          "Remaining Queue"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    n_amb = len(s.ambulance_wait_times)
    st.markdown("")
    if n_amb > 0:
        st.markdown(f"""
        <div class="alert-red">
        <strong>{n_amb} ambulance(s)</strong> served.
        Avg preemption wait: <strong>{avg_amb_wait:.1f}s</strong>
        vs car avg wait: <strong>{avg_car_wait:.1f}s</strong>
        — time saved: <strong>{max(0, avg_car_wait - avg_amb_wait):.1f}s</strong>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-green">No ambulances arrived in this run.</div>',
                    unsafe_allow_html=True)

    # ── ANIMATED VISUALIZATION ────────────────────
    st.markdown('<div class="section-header">Animated Simulation Playback</div>',
                unsafe_allow_html=True)

    if frames:
        col_anim, col_ctrl = st.columns([2, 1])

        with col_ctrl:
            st.markdown("**Playback Controls**")
            frame_idx = st.slider(
                "Frame", 0, len(frames) - 1,
                st.session_state.get("frame_idx", 0),
                key="frame_slider",
            )
            st.session_state["frame_idx"] = frame_idx

            play_col, stop_col = st.columns(2)
            play_btn = play_col.button("Play", use_container_width=True, type="primary")
            stop_btn = stop_col.button("Stop", use_container_width=True)

            if stop_btn:
                st.session_state["playing"] = False
            if play_btn:
                st.session_state["playing"] = True

            fr = frames[frame_idx]
            st.markdown("---")
            st.markdown(f"**Time:** `{fr['t']:.0f}s`")
            st.markdown(f"**Phase:** `{'NS Green' if fr['phase']==0 else 'EW Green'}`")
            if fr["emergency_active"]:
                st.markdown(f"**EMERGENCY:** `{fr['emergency_dir']} lane`")
            st.markdown(f"**Total arrivals:** `{fr['total_arrivals']}`")
            st.markdown(f"**Served:** `{fr['served']}`")
            st.markdown("---")
            st.markdown("**Queue sizes**")
            for d in DIRECTIONS:
                qlen = len(fr["queues"][d])
                bar  = "#" * min(qlen, 12) + "." * max(0, 12 - qlen)
                st.markdown(f"`{d[:1]}  [{bar}] {qlen}`")

        with col_anim:
            anim_placeholder = st.empty()

            # Draw current frame
            fig = draw_frame(frames[frame_idx], figsize=(6, 6))
            anim_placeholder.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Auto-play loop
            if st.session_state.get("playing", False):
                delay = 1.0 / anim_spd
                start = st.session_state.get("frame_idx", 0)
                for i in range(start, len(frames)):
                    if not st.session_state.get("playing", False):
                        break
                    fig = draw_frame(frames[i], figsize=(6, 6))
                    anim_placeholder.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.session_state["frame_idx"] = i
                    time.sleep(delay)
                st.session_state["playing"] = False
                st.rerun()

    # ── Queue history + phase chart ───────────────
    st.markdown('<div class="section-header">Queue History</div>', unsafe_allow_html=True)
    if s.queue_history:
        fig_q = plot_queue_history(s.queue_history)
        st.pyplot(fig_q, use_container_width=True)
        plt.close(fig_q)

    # ── Wait time distribution ─────────────────────
    st.markdown('<div class="section-header">Wait Time Distribution</div>', unsafe_allow_html=True)
    fig_w = plot_wait_distribution(s.car_wait_times, s.ambulance_wait_times)
    st.pyplot(fig_w, use_container_width=True)
    plt.close(fig_w)

    # ── Per-direction table ────────────────────────
    st.markdown('<div class="section-header">Per-Direction Summary</div>', unsafe_allow_html=True)
    dir_data = []
    for d in DIRECTIONS:
        dir_vehs = [v for v in s.served_vehicles if v.direction == d]
        dir_data.append({
            "Direction":   d,
            "Served":      len(dir_vehs),
            "Remaining":   len(s.queues[d]),
            "Avg Wait (s)": f"{np.mean([v.wait_time for v in dir_vehs]):.2f}" if dir_vehs else "—",
        })
    st.dataframe(pd.DataFrame(dir_data), hide_index=True, use_container_width=True)

    # ── Sensitivity analysis ───────────────────────
    if sens_btn:
        st.markdown('<div class="section-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
        with st.spinner("Running sensitivity sweep..."):
            fig_s = plot_sensitivity(params)
            st.pyplot(fig_s, use_container_width=True)
            plt.close(fig_s)
        st.markdown("""
        **Insight:** As ambulance probability increases, more emergency preemptions occur.
        Ambulance wait times stay low due to signal override, while car wait times
        increase slightly due to longer red periods caused by preemptions.
        """)

    # ── Export ─────────────────────────────────────
    st.markdown('<div class="section-header">Export Data</div>', unsafe_allow_html=True)
    if s.queue_history:
        st.download_button("Download Queue History CSV",
                           pd.DataFrame(s.queue_history).to_csv(index=False),
                           file_name="queue_history.csv", mime="text/csv")
    served_df = pd.DataFrame([{
        "VehicleID": v.vid, "Direction": v.direction, "Type": v.vehicle_type,
        "ArrivalTime": f"{v.arrival_time:.1f}", "WaitTime": f"{v.wait_time:.2f}",
        "DepartureTime": f"{v.departure_time:.1f}",
    } for v in s.served_vehicles])
    if not served_df.empty:
        st.download_button("Download Vehicle Log CSV",
                           served_df.to_csv(index=False),
                           file_name="vehicle_log.csv", mime="text/csv")

    # ── Footer ─────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='color:#4a5568;font-size:0.78rem;text-align:center;font-family:monospace;'>"
        "Stochastic Modeling &amp; Simulation Project · Poisson Arrivals · M/M/c Queue · "
        "Emergency Preemption · Universitas Gadjah Mada</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
