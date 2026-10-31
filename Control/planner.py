
#the map, A* path search and path smoothing

import heapq

import numpy as np
from scipy.ndimage import binary_dilation

import frames as F
from Build_scene import OBSTACLES, ROOM_X, ROOM_Y, WALL_THICKNESS
from Control.nav_settings import SAFETY_MARGIN


class OccupancyGrid:
    """
    the ward as a grid of cells: True = obstacle, False = free.
    occ[row, col] with rows along +y and columns along +x, the same layout as
    Person 1's lidar_map.OccupancyGrid (rows = iy, cols = ix)
    """

    def __init__(self, occ, resolution_m, origin=(0.0, 0.0)):
        self.occ = np.asarray(occ, dtype=bool)
        self.resolution_m = float(resolution_m)
        self.origin = np.asarray(origin, dtype=float)       #world xy of the corner of cell (0, 0)

    def world_to_cell(self, x, y):
        return (int(np.floor((y - self.origin[1]) / self.resolution_m)),
                int(np.floor((x - self.origin[0]) / self.resolution_m)))

    def cell_to_world(self, row, col):
        return (self.origin[0] + (col + 0.5) * self.resolution_m,
                self.origin[1] + (row + 0.5) * self.resolution_m)

    def in_bounds(self, row, col):
        return 0 <= row < self.occ.shape[0] and 0 <= col < self.occ.shape[1]

    def is_free(self, x, y):
        row, col = self.world_to_cell(x, y)
        return self.in_bounds(row, col) and not self.occ[row, col]

    def segment_free(self, p, q):
        #checks points every half cell along the line so no cell gets skipped
        length = np.hypot(q[0] - p[0], q[1] - p[1])
        n = max(2, int(np.ceil(length / (0.5 * self.resolution_m))) + 1)
        return all(self.is_free(p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1]))
                   for s in np.linspace(0, 1, n))

    def inflate(self, radius):
        #grow every obstacle by radius so the planner can treat the robot as a point
        #(configuration space)
        n = int(np.ceil(radius / self.resolution_m))
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
        disk = xx ** 2 + yy ** 2 <= n * n
        return OccupancyGrid(binary_dilation(self.occ, structure=disk),
                             self.resolution_m, self.origin)

    @property
    def extent(self):
        #(x_min, x_max, y_min, y_max), for plotting with matplotlib
        h, w = self.occ.shape
        return (self.origin[0], self.origin[0] + w * self.resolution_m,
                self.origin[1], self.origin[1] + h * self.resolution_m)


#map location

def build_grid_from_lidar(lidar_map):
    """
    convert Person 1's log-odds map into a planning grid.
    only cells the LiDAR is confident are occupied count as obstacles. unknown cells
    (never seen) count as free, which is optimistic: the robot replans if a scan
    later shows something in the way
    """
    occupied = lidar_map.classify() == 1
    return OccupancyGrid(occupied, lidar_map.res, lidar_map.origin)


def build_known_grid(resolution_m=0.05):
    #map drawn straight from Build_scene's obstacle list, for comparing against the lidar map
    x_min, y_min = -0.2, -0.2
    nx = int(np.ceil((ROOM_X + 0.4) / resolution_m))
    ny = int(np.ceil((ROOM_Y + 0.4) / resolution_m))
    occ = np.zeros((ny, nx), dtype=bool)
    walls = [
        ("wall_S", -WALL_THICKNESS, -WALL_THICKNESS, ROOM_X + WALL_THICKNESS, 0.0),
        ("wall_N", -WALL_THICKNESS, ROOM_Y, ROOM_X + WALL_THICKNESS, ROOM_Y + WALL_THICKNESS),
        ("wall_W", -WALL_THICKNESS, 0.0, 0.0, ROOM_Y),
        ("wall_E", ROOM_X, 0.0, ROOM_X + WALL_THICKNESS, ROOM_Y),
    ]
    #note Build_scene orders boxes as (name, x_min, y_min, x_max, y_max, height)
    for name, x0, y0, x1, y1, *_ in walls + OBSTACLES:
        c0 = max(0, int(np.floor((x0 - x_min) / resolution_m)))
        c1 = min(nx, int(np.ceil((x1 - x_min) / resolution_m)))
        r0 = max(0, int(np.floor((y0 - y_min) / resolution_m)))
        r1 = min(ny, int(np.ceil((y1 - y_min) / resolution_m)))
        occ[r0:r1, c0:c1] = True
    return OccupancyGrid(occ, resolution_m, (x_min, y_min))


#path searching
def astar(grid, start_xy, goal_xy):
    """
    A* search: shortest path through free cells, moving in 8 directions.
    always expands the cell with the lowest (cost so far + straight line distance
    still to go). straight line never overestimates, so the path is the shortest
    """
    start, goal = grid.world_to_cell(*start_xy), grid.world_to_cell(*goal_xy)
    for label, cell in (("start", start), ("goal", goal)):
        if not grid.in_bounds(*cell) or grid.occ[cell]:
            raise ValueError(f"The {label} {np.round(start_xy if label == 'start' else goal_xy, 2)} "
                             "is outside the map or too close to an obstacle")

    moves = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
             (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    def distance_to_go(c):
        return np.hypot(c[0] - goal[0], c[1] - goal[1])

    queue = [(distance_to_go(start), 0.0, start)]       #(priority, cost so far, cell)
    came_from, cost = {start: None}, {start: 0.0}
    while queue:
        _, cost_so_far, cell = heapq.heappop(queue)
        if cell == goal:
            break
        if cost_so_far > cost[cell]:
            continue                                     #stale queue entry
        for dr, dc, step in moves:
            neighbour = (cell[0] + dr, cell[1] + dc)
            if not grid.in_bounds(*neighbour) or grid.occ[neighbour]:
                continue
            if dr and dc and (grid.occ[cell[0] + dr, cell[1]] or grid.occ[cell[0], cell[1] + dc]):
                continue                                 #dont squeeze between diagonal obstacles
            if cost_so_far + step < cost.get(neighbour, np.inf):
                cost[neighbour], came_from[neighbour] = cost_so_far + step, cell
                heapq.heappush(queue, (cost_so_far + step + distance_to_go(neighbour),
                                       cost_so_far + step, neighbour))
    else:
        raise RuntimeError("A* found no path")

    cells = []                                           #walk back from goal to start
    while cell is not None:
        cells.append(cell)
        cell = came_from[cell]
    path = np.array([grid.cell_to_world(r, c) for r, c in reversed(cells)])
    path[0], path[-1] = start_xy, goal_xy
    return path


def shortcut(grid, path):
    #remove waypoints wherever a straight line between them is clear
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not grid.segment_free(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return np.array(out)


def chaikin(points, iterations=3):
    #round off corners: swap each corner for points 1/4 and 3/4 along its sides, repeat
    p = np.asarray(points, dtype=float)
    for _ in range(iterations):
        q = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        q.append(p[-1])
        p = np.array(q)
    return p


def plan_to_bedside(grid, start_pose, park_pose, approach_distance=0.8, lead_in=0.0):
    """
    plan from start_pose to park_pose, ending with a straight head-on approach.

    head-on matters: a stopped robot cant fix a sideways error (controllability
    drops to rank 2), so sideways error has to be gone before it slows down.

    lead_in: when replanning mid-drive, the path starts with a short straight bit
    along the robots current heading so the ghost doesnt ask for a sudden turn
    """
    park_x, park_y, park_theta = park_pose
    approach_xy = np.array([park_x - approach_distance * np.cos(park_theta),
                            park_y - approach_distance * np.sin(park_theta)])
    planning_grid = grid.inflate(F.ROBOT_RADIUS + SAFETY_MARGIN)
    body_grid = grid.inflate(F.ROBOT_RADIUS)

    start_xy = np.asarray(start_pose[:2], dtype=float)
    head = [start_xy]
    if lead_in > 0:
        ahead_xy = start_xy + lead_in * np.array([np.cos(start_pose[2]), np.sin(start_pose[2])])
        if planning_grid.is_free(*ahead_xy) and body_grid.segment_free(start_xy, ahead_xy):
            head.append(ahead_xy)

    #if the robot has drifted into the safety margin, plan from the nearest free cell
    search_from = _nearest_free(planning_grid, head[-1])
    if np.linalg.norm(search_from - head[-1]) > 1e-9:
        head.append(search_from)

    raw_astar_path = astar(planning_grid, search_from, approach_xy)
    waypoints = np.vstack([np.reshape(head[:-1], (-1, 2)), shortcut(planning_grid, raw_astar_path),
                           [[park_x, park_y]]])
    #smoothing cuts corners, so check the smoothed path again. if a corner now clips
    #something, smooth less (3 rounds, then 2, 1, 0 = the straight-line route, which is clear)
    for rounds in (3, 2, 1, 0):
        smooth_path = chaikin(waypoints, rounds)
        clash = next(((a, b) for a, b in zip(smooth_path[:-1], smooth_path[1:])
                      if not body_grid.segment_free(a, b)), None)
        if clash is None:
            return smooth_path
    raise RuntimeError(f"Route clips an obstacle between {np.round(clash[0], 2)} and {np.round(clash[1], 2)}")


def _nearest_free(grid, xy, max_search_m=0.5):
    #closest free cell to xy (xy itself if its already free)
    if grid.is_free(*xy):
        return np.asarray(xy, dtype=float)
    r0, c0 = grid.world_to_cell(*xy)
    n = int(max_search_m / grid.resolution_m)
    best, best_d = None, np.inf
    for dr in range(-n, n + 1):
        for dc in range(-n, n + 1):
            r, c = r0 + dr, c0 + dc
            if grid.in_bounds(r, c) and not grid.occ[r, c] and dr * dr + dc * dc < best_d:
                best, best_d = (r, c), dr * dr + dc * dc
    if best is None:
        raise RuntimeError("Robot is boxed in: no free cell within 0.5 m")
    return np.array(grid.cell_to_world(*best))


def path_blocked(grid, path_xy):
    #true if any part of the path now runs into something (robot radius only, no margin)
    body_grid = grid.inflate(F.ROBOT_RADIUS)
    return any(not body_grid.segment_free(a, b) for a, b in zip(path_xy[:-1], path_xy[1:]))
