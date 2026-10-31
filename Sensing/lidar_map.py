# Log-odds occupancy grid, every 5cm cell stores l = log(p/(1-p))

    # cells the beam passed THROUGH  -> evidence of free space : l += l_free (< 0)
    # the cell the beam ENDED in     -> evidence of an obstacle: l += l_occ  (> 0)
    # a beam with no return (inf)    -> free all the way to max range, no hit
    # a NaN beam                     -> ignored

# clamping l to +/-1_clap keeps the map adaptable incase scene changes (e.g. trolley moves)

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.ndimage import distance_transform_edt

import frames as F
from Build_scene import ROOM_X, ROOM_Y      # 6 m x 4 m


class OccupancyGrid:
    def __init__(self, x_min=-0.2, y_min=-0.2, x_max=ROOM_X + 0.2, y_max=ROOM_Y + 0.2,
                 res=0.05, l_occ=0.85, l_free=-0.4, l_clamp=5.0, max_range=3.5):
        
        # res     : cell size [m]. 5 cm is fine enough for the 0.75 m corridor between the beds.
        # l_occ   : log-odds added for a hit. 0.85 corresponds to p ~ 0.7 per scan.
        # l_free  : log-odds added for a pass-through. -0.4 corresponds to p ~ 0.4, deliberately weaker than l_occ, because a beam can graze past thin obstacles.
        
        self.origin = np.array([x_min, y_min], dtype=float)
        self.res = res
        self.nx = int(np.ceil((x_max - x_min) / res))
        self.ny = int(np.ceil((y_max - y_min) / res))
        self.L = np.zeros((self.ny, self.nx))
        self.l_occ, self.l_free, self.l_clamp = l_occ, l_free, l_clamp
        self.max_range = max_range
        # Samples along each beam every half cell, so the beam does not skip cells.
        self._s = np.arange(0.0, max_range, res / 2)

    # indexing
    def world_to_cell(self, xy):                        # Convert (N, 2) world points to (ix, iy) integer arrays (may be out of bounds).
        ij = np.floor((np.atleast_2d(xy) - self.origin) / self.res).astype(int)
        return ij[:, 0], ij[:, 1]

    def cell_to_world(self, ix, iy):                #Return the world coordinates of the centres of cells (ix, iy)
        return self.origin + (np.stack([ix, iy], axis=-1) + 0.5) * self.res

    def extent(self):
        x0, y0 = self.origin
        return [x0, x0 + self.nx * self.res, y0, y0 + self.ny * self.res]

    def _flat(self, ix, iy):
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return iy[ok] * self.nx + ix[ok]

    # ------------------------------------------------------------ update ---
    def update(self, angles, ranges, sensor_pose):
        
        # sensor_pose = world pose of the LIDAR frame L, from frames.lidar_pose(base_pose)
        # The LDS-01 is 6.4 cm behind B, ignoring would smear every wall
        
        xs, ys, th = sensor_pose
        valid = ~np.isnan(ranges)
        a = angles[valid] + th
        r = ranges[valid]
        hit = np.isfinite(r)
        r_free = np.where(hit, np.minimum(r, self.max_range), self.max_range)
        c, s = np.cos(a), np.sin(a)

        # Free space: every sample along each beam, stopping half a cell short of the hit.
        mask = self._s[None, :] < (r_free[:, None] - self.res / 2)
        px = xs + c[:, None] * self._s[None, :]
        py = ys + s[:, None] * self._s[None, :]
        ix = np.floor((px[mask] - self.origin[0]) / self.res).astype(int)
        iy = np.floor((py[mask] - self.origin[1]) / self.res).astype(int)
        free = np.unique(self._flat(ix, iy))   # each cell updated at most once per scan

        # Hits: the end cell of each beam that returned.
        hx = xs + c[hit] * r[hit]
        hy = ys + s[hit] * r[hit]
        occ = np.unique(self._flat(np.floor((hx - self.origin[0]) / self.res).astype(int),
                                   np.floor((hy - self.origin[1]) / self.res).astype(int)))
        free = np.setdiff1d(free, occ, assume_unique=True)  # a hit outranks a pass-through

        Lf = self.L.reshape(-1)          # a view, so in-place edits change self.L
        Lf[free] += self.l_free
        Lf[occ] += self.l_occ
        np.clip(self.L, -self.l_clamp, self.l_clamp, out=self.L)

    # ------------------------------------------------------------ outputs ---
    def probability(self):                  #Return p(occupied) for every cell: p = 1 - 1 / (1 + e^l)
        return 1.0 - 1.0 / (1.0 + np.exp(self.L))

    def classify(self, occ_thresh=0.65, free_thresh=0.35):              #Return an int8 grid: 1 = occupied, 0 = free, -1 = unknown (never observed, or ambiguous)
        p = self.probability()
        out = -np.ones(self.L.shape, dtype=np.int8)
        out[p >= occ_thresh] = 1
        out[p <= free_thresh] = 0
        return out

    def blocked(self, radius=F.ROBOT_RADIUS, unknown_as_occupied=False):            
        
        # returns configuration-space obstacle grid, True = robot centre NOT go here
        # distance_transform_edt gives each cell's distance to the nearest occupied cell.
       
        # unknown_as_occupied=False is optimistic: the planner may cross cells never seen. That is necessary when the map comes from one partial run.
        # Make it True for the "safe" variant in the robustness study.
        
        cls = self.classify()
        occ = cls == 1
        if unknown_as_occupied:
            occ |= cls == -1
        if radius <= 0 or not occ.any():
            return occ
        return distance_transform_edt(~occ) * self.res <= radius

    def to_planner(self, radius=F.ROBOT_RADIUS, unknown_as_occupied=False):             # interface handed to person 3 planner
        return {"blocked": self.blocked(radius, unknown_as_occupied),
                "prob": self.probability(),
                "origin": self.origin.copy(),   # world xy of the corner of cell (0, 0)
                "res": self.res,
                "shape": (self.ny, self.nx)}    # (rows = iy, cols = ix)

    @classmethod
    def from_arrays(cls, L, origin, res, **kw):                 #Rebuild a grid from a saved log (see record_run.py)
        
        ny, nx = L.shape
        g = cls(x_min=origin[0], y_min=origin[1],
                x_max=origin[0] + nx * res, y_max=origin[1] + ny * res, res=float(res), **kw)
        g.L = np.array(L, dtype=float)
        g.ny, g.nx = ny, nx          # trust the saved shape over float rounding
        return g
