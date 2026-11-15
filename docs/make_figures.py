"""README animations for the RoboRacer stack, drawn on our real competition map.

    python docs/make_figures.py

They are illustrations: a simulated car and LiDAR on roboracer/maps/competition_track.pgm, running simple
re-implementations of each stage of our stack (occupancy mapping as Cartographer does it, a particle filter,
the adaptive pure pursuit from roboracer/controller/adaptive_pure_pursuit.py, follow-the-gap, and the mux).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_sim import Track, wrap   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BG, FG, MUTED = "#05070a", "#f3f4f6", "#9ca3af"
RED, CYAN, GREEN, AMBER = "#ef4444", "#22d3ee", "#22c55e", "#f59e0b"
T = Track()
rng = np.random.default_rng(3)
R0, R1, C0, C1 = 125, 300, 120, 330          # view window (rows / cols of the 2.4 cm grid)


def grab(fig, frames):
    fig.canvas.draw()
    frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(colors=255))


def base_map(occ=None, free=None):
    occ = T.occ if occ is None else occ
    free = T.free if free is None else free
    img = np.zeros(occ.shape + (3,))
    img[:] = (0.07, 0.08, 0.10)                     # unknown
    img[free] = (0.17, 0.19, 0.23)                  # free space
    img[occ] = (0.93, 0.94, 0.96)                   # walls
    return img


def car_patch(r, c, yaw, size=9, col=CYAN):
    pts = np.array([[size, 0], [-size * 0.6, size * 0.55], [-size * 0.6, -size * 0.55]])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    xy = pts @ R.T
    return Polygon(np.c_[c + xy[:, 0], r - xy[:, 1]], closed=True, fc=col, ec="white", lw=1.2, zorder=6)


def path_poses(step_m=0.05, laps=1.3):
    """Poses (row, col, yaw) along the raceline, one every step_m metres."""
    race = np.r_[T.race, T.race[:1]]
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(race, axis=0).T))] * T.res
    s = np.arange(0, seg[-1] * laps, step_m) % seg[-1]
    r = np.interp(s, seg, race[:, 0]); c = np.interp(s, seg, race[:, 1])
    dr, dc = np.gradient(r), np.gradient(c)
    return np.c_[r, c, np.arctan2(-dr, dc)]


# ====================================================================================== 1. mapping
def mapping_gif():
    """Occupancy-grid mapping from scans: what Cartographer's 2-D submaps accumulate as the car drives."""
    poses = path_poses(0.06, 1.15)
    logodds = np.zeros((T.H, T.W))
    fig = plt.figure(figsize=(8, 6.4), dpi=80, facecolor=BG)
    frames, dur = [], []
    for i, (r, c, yaw) in enumerate(poses):
        ang, ranges, (ri, ci, first, inside) = T.scan((r, c, yaw), n_beams=240, max_range=6.0)
        for b in range(len(ang)):
            k = first[b]
            ok = inside[b, :k]
            logodds[ri[b, :k][ok], ci[b, :k][ok]] -= 0.35
            if inside[b, k] and ranges[b] < 6.0:
                logodds[ri[b, k], ci[b, k]] += 1.2
        np.clip(logodds, -3, 4, out=logodds)
        if i % 2:
            continue
        fig.clf(); fig.set_facecolor(BG)
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.86]); ax.set_facecolor(BG); ax.axis("off")
        img = np.full((T.H, T.W, 3), 0.07)
        img[logodds < -0.5] = (0.17, 0.19, 0.23)
        img[logodds > 1.0] = (0.93, 0.94, 0.96)
        ax.imshow(img, interpolation="nearest")
        th = yaw + ang
        ax.plot(c + np.cos(th) * ranges / T.res, r - np.sin(th) * ranges / T.res, ".", color=RED, ms=2.2)
        for b in range(0, len(ang), 12):
            ax.plot([c, c + np.cos(th[b]) * ranges[b] / T.res], [r, r - np.sin(th[b]) * ranges[b] / T.res],
                    color=RED, lw=0.4, alpha=0.35)
        trail = poses[:i + 1]
        ax.plot(trail[:, 1], trail[:, 0], color=CYAN, lw=1.5, alpha=0.8)
        ax.add_patch(car_patch(r, c, yaw))
        ax.set_xlim(C0 - 30, C1 + 30); ax.set_ylim(R1 + 30, R0 - 40)
        known = (np.abs(logodds) > 0.5).sum() * T.res ** 2
        fig.text(0.03, 0.93, "Mapping · the occupancy grid grows scan by scan (Cartographer, 2-D)", color=FG,
                 fontsize=13, weight="bold")
        fig.text(0.03, 0.895, f"area mapped {known:5.1f} m²   ·   red: current LiDAR scan (270°)", color=MUTED,
                 fontsize=10.5)
        grab(fig, frames); dur.append(90)
    dur[-1] = 2500
    frames[0].save(OUT / "mapping.gif", save_all=True, append_images=frames[1:], duration=dur, loop=0)
    plt.close(fig)


# ====================================================================================== 2. particle filter
def pf_gif(n=3000, beams=24):
    """Global localisation: particles start everywhere on the map and collapse onto the car."""
    poses = path_poses(0.05, 1.6)
    free_r, free_c = np.nonzero(T.free)
    k = rng.integers(0, len(free_r), n)
    P = np.c_[free_r[k] + rng.random(n), free_c[k] + rng.random(n), rng.uniform(-np.pi, np.pi, n)]
    sigma = 0.15
    temper = 0.12          # soften the likelihood so convergence takes several scans (as with real noise)
    fig = plt.figure(figsize=(8, 6.4), dpi=80, facecolor=BG)
    frames, dur = [], []
    view = [C0 - 100, C1 + 50, R1 + 110, R0 - 100]
    for i in range(1, len(poses)):
        prev, cur = poses[i - 1], poses[i]
        # ---- motion model: odometry in the robot frame + noise
        dx_w, dy_w = (cur[1] - prev[1]), -(cur[0] - prev[0])
        fwd = dx_w * np.cos(prev[2]) + dy_w * np.sin(prev[2])
        side = -dx_w * np.sin(prev[2]) + dy_w * np.cos(prev[2])
        dyaw = wrap(cur[2] - prev[2])
        f = fwd * (1 + 0.1 * rng.standard_normal(n)); s = side + 0.15 * rng.standard_normal(n)
        P[:, 2] = wrap(P[:, 2] + dyaw + 0.03 * rng.standard_normal(n))
        P[:, 1] += f * np.cos(P[:, 2]) - s * np.sin(P[:, 2])
        P[:, 0] -= f * np.sin(P[:, 2]) + s * np.cos(P[:, 2])
        # ---- sensor model: likelihood field (distance from each beam end-point to the nearest wall)
        ang, ranges, _ = T.scan(tuple(cur), n_beams=beams, max_range=6.0)
        use = ranges < 5.9
        th = P[:, 2:3] + ang[use][None, :]
        er = (P[:, 0:1] - np.sin(th) * ranges[use][None, :] / T.res).round().astype(int)
        ec = (P[:, 1:2] + np.cos(th) * ranges[use][None, :] / T.res).round().astype(int)
        ok = (er >= 0) & (er < T.H) & (ec >= 0) & (ec < T.W)
        d = np.full(er.shape, 1.0)
        d[ok] = T.dist[er[ok], ec[ok]]
        logw = temper * np.log(0.8 * np.exp(-0.5 * (np.minimum(d, 0.6) / sigma) ** 2) + 0.2 * 0.05).sum(1)
        pr, pc = P[:, 0].round().astype(int), P[:, 1].round().astype(int)
        valid = (pr >= 0) & (pr < T.H) & (pc >= 0) & (pc < T.W)
        valid[valid] = T.free[pr[valid], pc[valid]]
        logw[~valid] = -1e9
        w = np.exp(logw - logw.max()); w /= w.sum()
        est = P[:, :2].mean(0)
        spread = np.sqrt(((P[:, :2] - est) ** 2).sum(1).mean()) * T.res      # spread of the cloud being drawn
        err = np.hypot(*(est - prev[:2])) * T.res
        # ---- draw the cloud as it was before this update (every step early on, then every other step)
        if i < 40 or i % 2 == 0:
            fig.clf(); fig.set_facecolor(BG)
            ax = fig.add_axes([0.02, 0.02, 0.96, 0.84]); ax.set_facecolor(BG); ax.axis("off")
            ax.imshow(base_map(), interpolation="nearest")
            ax.plot(P[:, 1], P[:, 0], ".", color=RED, ms=1.8, alpha=0.55)
            ax.add_patch(car_patch(prev[0], prev[1], prev[2], col=CYAN))
            if spread < 0.6:
                ax.add_patch(Circle((est[1], est[0]), 6, fill=False, ec=GREEN, lw=2, zorder=7))
            for v in (view,):
                ax.set_xlim(v[0], v[1]); ax.set_ylim(v[2], v[3])
            fig.text(0.03, 0.93, "Localisation · particle filter on the race map", color=FG, fontsize=13, weight="bold")
            fig.text(0.03, 0.895, f"{n} particles   ·   spread {spread:5.2f} m   ·   error {err:5.2f} m"
                     + ("   ·   locked ✓" if spread < 0.15 and err < 0.15 else ""),
                     color=GREEN if (spread < 0.15 and err < 0.15) else MUTED,
                     fontsize=10.5, family="monospace")
            grab(fig, frames); dur.append(1500 if i == 1 else (160 if i < 40 else 100))
        # ---- resample (low variance) + small jitter
        u = (rng.random() + np.arange(n)) / n
        idx = np.minimum(np.searchsorted(np.cumsum(w), u), n - 1)
        P = P[idx] + np.c_[0.6 * rng.standard_normal(n), 0.6 * rng.standard_normal(n), 0.02 * rng.standard_normal(n)]
    dur[-1] = 2500
    frames[0].save(OUT / "particle_filter.gif", save_all=True, append_images=frames[1:], duration=dur, loop=0)
    plt.close(fig)


# ====================================================================================== 3. ALPP + FTG + mux
def alpp_gif():
    L_wb = 0.33
    la_min, la_max = 0.35, 0.75           # look-ahead band (scaled to this small track; 1.0–2.0 m on the big one)
    v_min, v_max = 0.3, 2.2
    race = T.race
    race_m = np.c_[race[:, 1], -race[:, 0]] * T.res          # metric x (right), y (up)
    # obstacle: a 25 cm box sitting on the raceline
    occ = T.occ.copy()
    ob = race[int(len(race) * 0.55)]
    rr, cc = np.mgrid[-5:6, -5:6]
    occ[(ob[0] + rr).astype(int), (ob[1] + cc).astype(int)] = True
    x, y = race_m[0]; d = race_m[1] - race_m[0]; yaw = np.arctan2(d[1], d[0]); v = 0.5
    dt, beta, area_win, prev_dev = 0.04, 0.5, [], 0.0
    mode, clear_count = "ALPP", 0
    hist = []
    fig = plt.figure(figsize=(10, 6), dpi=80, facecolor=BG)
    frames, dur = [], []
    for step in range(int(2.3 * T.lap_m / 1.2 / dt)):
        pose_rc = (-y / T.res, x / T.res, yaw)
        ang, ranges, _ = T.scan(pose_rc, n_beams=181, fov=np.deg2rad(180), max_range=4.0, occ=occ)
        # ---------------- mux: LiDAR points that are NOT in the stored map, close ahead -> new obstacle -> FTG
        th_w = yaw + ang
        er = np.clip((-y - np.sin(th_w) * ranges) / T.res, 0, T.H - 1).round().astype(int)
        ec = np.clip((x + np.cos(th_w) * ranges) / T.res, 0, T.W - 1).round().astype(int)
        unexpected = (T.dist[er, ec] > 0.12) & (ranges < 3.9)
        threat = unexpected & (np.abs(ang) < np.deg2rad(35)) & (ranges < 1.0)
        if mode == "ALPP" and threat.any():
            mode, clear_count = "FTG", 0
        elif mode == "FTG":
            clear_count = 0 if (unexpected & (np.abs(ang) < np.deg2rad(60)) & (ranges < 0.9)).any() else clear_count + 1
            if clear_count > 8:
                mode = "ALPP"
        # ---------------- ALPP (adaptive look-ahead pure pursuit, as in adaptive_pure_pursuit.py)
        sig = 1 / (1 + np.exp(-(((v - v_min) / (v_max - v_min)) * 10 - 5)))
        la = la_min + sig * (la_max - la_min)
        dists = np.hypot(race_m[:, 0] - x, race_m[:, 1] - y)
        ci = int(np.argmin(dists))
        gi = ci
        while np.hypot(*(race_m[gi % len(race_m)] - [x, y])) < la:
            gi += 1
        goal = race_m[gi % len(race_m)]
        dx, dy = goal[0] - x, goal[1] - y
        alpha = wrap(np.arctan2(dy, dx) - yaw)
        dev = dists[ci]
        area_win.append((dev + prev_dev) / 2 * max(v * dt, 1e-3)); area_win = area_win[-10:]; prev_dev = dev
        area = sum(area_win)
        kappa = 2 * np.sin(alpha) / la
        v_curv = min(v_max, np.sqrt(1.6 / abs(kappa))) if abs(kappa) > 1e-6 else v_max
        v_dev = v_max / (1 + 8 * area)
        beta = min(1.0, beta + 0.25) if area < 0.02 else (max(0.0, beta - 0.25) if v < 0.3 * v_max else beta)
        if mode == "ALPP":
            steer = np.arctan2(2 * L_wb * np.sin(alpha), la)
            v_cmd = beta * v_curv + (1 - beta) * v_dev
            gap_dir = None
        else:
            # ---------------- follow the gap: bubble around the closest point, steer to the widest gap
            rg = ranges.copy()
            k0 = int(np.argmin(rg))
            bub = int(np.degrees(np.arctan2(0.3, max(rg[k0], 0.05))))
            rg[max(0, k0 - bub):k0 + bub + 1] = 0
            ok = rg > 0.9
            best, bs, cur_s = (0, 0), 0, None
            for k in range(len(ok) + 1):
                if k < len(ok) and ok[k]:
                    cur_s = k if cur_s is None else cur_s
                elif cur_s is not None:
                    if k - cur_s > bs:
                        best, bs = (cur_s, k - 1), k - cur_s
                    cur_s = None
            gap_dir = ang[(best[0] + best[1]) // 2]
            steer = np.clip(gap_dir, -0.4, 0.4)
            v_cmd = 0.9
        steer = float(np.clip(steer, -0.42, 0.42))
        v += np.clip(v_cmd - v, -4 * dt, 4 * dt)
        x += v * np.cos(yaw) * dt; y += v * np.sin(yaw) * dt; yaw = wrap(yaw + v / L_wb * np.tan(steer) * dt)
        hist.append((x, y, v, mode))
        if step % 3:
            continue
        fig.clf(); fig.set_facecolor(BG)
        ax = fig.add_axes([0.01, 0.03, 0.6, 0.84]); ax.set_facecolor(BG); ax.axis("off")
        ax.imshow(base_map(occ=occ), interpolation="nearest")
        ax.plot(race[:, 1], race[:, 0], "--", color=GREEN, lw=1.2, alpha=0.8)
        H = np.array(hist)
        hx, hy, hv = H[:, 0].astype(float), H[:, 1].astype(float), H[:, 2].astype(float)
        ax.scatter(hx / T.res, -hy / T.res, c=hv, cmap="plasma", vmin=0, vmax=v_max, s=6, zorder=4)
        pr, pc = -y / T.res, x / T.res
        if mode == "ALPP":
            ax.add_patch(Circle((pc, pr), la / T.res, fill=False, ec=CYAN, lw=1, ls=":", zorder=5))
            ax.plot(goal[0] / T.res, -goal[1] / T.res, "o", color=CYAN, ms=7, zorder=6)
            ax.plot([pc, goal[0] / T.res], [pr, -goal[1] / T.res], color=CYAN, lw=1, zorder=5)
        else:
            th = yaw + gap_dir
            ax.annotate("", (pc + np.cos(th) * 40, pr - np.sin(th) * 40), (pc, pr),
                        arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.2), zorder=7)
            thb = yaw + ang
            ax.plot(pc + np.cos(thb) * ranges / T.res, pr - np.sin(thb) * ranges / T.res, ".", color=AMBER, ms=1.6)
        ax.add_patch(car_patch(pr, pc, yaw, size=8, col=CYAN if mode == "ALPP" else AMBER))
        ax.set_xlim(C0 - 5, C1 + 5); ax.set_ylim(R1 + 5, R0 - 15)
        # side panel
        s = fig.add_axes([0.64, 0.5, 0.33, 0.3]); s.set_facecolor(BG)
        s.plot(np.arange(len(hv)) * dt, hv, color=FG, lw=1.4)
        modes = np.array([m == "FTG" for m in H[:, 3]])
        s.fill_between(np.arange(len(hv)) * dt, 0, v_max * 1.1, where=modes, color=AMBER, alpha=0.25, lw=0)
        s.set_ylim(0, v_max * 1.1); s.set_xlim(0, max(4, len(hv) * dt))
        s.tick_params(colors=MUTED, labelsize=8); [sp.set_color("#374151") for sp in s.spines.values()]
        s.set_title("speed (m/s) · amber = FTG active", color=MUTED, fontsize=9.5, loc="left")
        badge = "ALPP  ·  racing line" if mode == "ALPP" else "FTG  ·  obstacle avoidance"
        fig.text(0.64, 0.33, "mux output →", color=MUTED, fontsize=11)
        fig.text(0.64, 0.26, badge, color=CYAN if mode == "ALPP" else AMBER, fontsize=15, weight="bold")
        fig.text(0.64, 0.17, f"look-ahead {la:4.2f} m   speed {v:4.2f} m/s" if mode == "ALPP"
                 else f"steering to widest gap {np.degrees(gap_dir):+5.1f}°", color=FG, fontsize=11, family="monospace")
        fig.text(0.02, 0.92, "Control · adaptive pure pursuit on the raceline, follow-the-gap around obstacles",
                 color=FG, fontsize=13, weight="bold")
        grab(fig, frames); dur.append(90)
    dur[-1] = 2000
    frames[0].save(OUT / "alpp_ftg_mux.gif", save_all=True, append_images=frames[1:], duration=dur, loop=0)
    plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1:] or ["mapping", "pf", "alpp"]
    if "mapping" in what:
        mapping_gif()
    if "pf" in what:
        pf_gif()
    if "alpp" in what:
        alpp_gif()
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:28s} {f.stat().st_size / 1e6:6.2f} MB")
