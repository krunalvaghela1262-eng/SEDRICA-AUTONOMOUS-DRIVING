"""Small 2-D simulator on our RoboRacer competition map, used only to draw the README animations.

It ray-casts a LiDAR on the real occupancy grid (roboracer/maps/competition_track.pgm), extracts the track
centre line as a raceline, and provides a kinematic bicycle model. The algorithms animated with it
(occupancy mapping, particle filter, adaptive pure pursuit, follow-the-gap, mux) are re-implemented in
docs/make_figures.py following the parameters of our ROS 2 nodes.
"""
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DOWN = 2                                   # work on a 2x down-sampled grid: 2.4 cm cells


class Track:
    def __init__(self):
        raw = np.array(__import__("PIL.Image", fromlist=["Image"]).open(ROOT / "roboracer/maps/competition_track.pgm"))
        res0 = 0.012
        h, w = raw.shape
        raw = raw[: h - h % DOWN, : w - w % DOWN]
        blocks = raw.reshape(raw.shape[0] // DOWN, DOWN, raw.shape[1] // DOWN, DOWN)
        self.occ = (blocks == 0).any(axis=(1, 3))
        self.free = (blocks == 254).all(axis=(1, 3)) & ~self.occ
        self.unknown = ~self.occ & ~self.free
        self.res = res0 * DOWN
        self.H, self.W = self.occ.shape
        # likelihood field: distance (m) from every cell to the nearest wall
        self.dist = ndimage.distance_transform_edt(~self.occ) * self.res
        self._raceline()

    # ------------------------------------------------------------------ geometry
    def _raceline(self):
        """Centre line of the ring between the inner island and the outer wall."""
        lab, n = ndimage.label(self.occ)
        sizes = ndimage.sum(self.occ, lab, range(1, n + 1))
        # inner island = a compact obstacle component near the middle of the map
        cy, cx = self.H / 2, self.W / 2
        best, bd = None, 1e9
        for k in range(1, n + 1):
            if sizes[k - 1] < 30:
                continue
            yy, xx = np.nonzero(lab == k)
            d = np.hypot(yy.mean() - cy, xx.mean() - cx)
            if d < bd and (np.ptp(yy) < self.H * 0.25):
                best, bd = k, d
        iy, ix = np.nonzero(lab == best)
        c = np.array([iy.mean(), ix.mean()])
        # the driving ring = the free-space component that encloses the island but not the whole map
        flab, fn = ndimage.label(self.free)
        cand = {}
        for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            for rad in range(20, 120, 4):
                p = (c + rad * np.array([-np.sin(th), np.cos(th)])).round().astype(int)
                if 0 <= p[0] < self.H and 0 <= p[1] < self.W and flab[p[0], p[1]]:
                    cand[flab[p[0], p[1]]] = cand.get(flab[p[0], p[1]], 0) + 1
                    break
        ring = flab == max(cand, key=cand.get)
        self.ring = ring
        pts = []
        for th in np.linspace(0, 2 * np.pi, 360, endpoint=False):
            dvec = np.array([-np.sin(th), np.cos(th)])
            r = np.arange(1, max(self.H, self.W))
            p = (c[:, None] + dvec[:, None] * r).round().astype(int)
            ok = (p[0] >= 0) & (p[0] < self.H) & (p[1] >= 0) & (p[1] < self.W)
            p, rr = p[:, ok], r[ok]
            inring = ring[p[0], p[1]]
            if not inring.any():
                continue
            k = np.nonzero(inring)[0]
            clear = self.dist[p[0, k], p[1, k]]
            pts.append(c + dvec * rr[k[np.argmax(clear)]])             # max clearance from both walls along this ray
        pts = np.array(pts)
        # smooth (periodic low-pass) and resample at equal spacing
        F = np.fft.rfft(pts, axis=0); F[12:] = 0
        pts = np.fft.irfft(F, n=len(pts), axis=0)
        seg = np.r_[0, np.cumsum(np.hypot(*np.diff(np.r_[pts, pts[:1]], axis=0).T))]
        L = seg[-1]
        s = np.linspace(0, L, int(L * self.res / 0.1), endpoint=False)          # one point every 10 cm
        closed = np.r_[pts, pts[:1]]
        self.race = np.c_[np.interp(s, seg, closed[:, 0]), np.interp(s, seg, closed[:, 1])]   # (row, col)
        self.island = c
        self.lap_m = L * self.res

    # ------------------------------------------------------------------ sensing
    def scan(self, pose, n_beams=180, fov=np.deg2rad(270), max_range=10.0, occ=None):
        """Ray-cast a LiDAR at pose (row, col, yaw). Returns beam angles (rel.), ranges (m), and the cells each
        beam passed through (for mapping)."""
        occ = self.occ if occ is None else occ
        r0, c0, yaw = pose
        ang = np.linspace(-fov / 2, fov / 2, n_beams)
        K = int(max_range / self.res)
        steps = np.arange(1, K + 1)
        th = yaw + ang
        rr = r0 - np.sin(th)[:, None] * steps          # row axis points down: "up" on screen is -row
        cc = c0 + np.cos(th)[:, None] * steps
        ri, ci = rr.round().astype(int), cc.round().astype(int)
        inside = (ri >= 0) & (ri < self.H) & (ci >= 0) & (ci < self.W)
        hit = np.zeros_like(inside)
        hit[inside] = occ[ri[inside], ci[inside]]
        hit |= ~inside
        first = np.where(hit.any(1), hit.argmax(1), K - 1)
        ranges = (first + 1) * self.res
        return ang, ranges, (ri, ci, first, inside)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
