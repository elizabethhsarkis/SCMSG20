#the map, A* path search and path smoothing

import heapq

import numpy as np
from scipy.ndimage import binary_dilation

from config import ROBOT_RADIUS, SAFETY_MARGIN, WARD_BOXES, WARD_HEIGHT, WARD_WIDTH


class OccupancyGrid:
    #the ward as a grid of cells: True = obstacle, False = free
    #occ[row, col], with rows along +y and columns along +x

    def __init__(self, occ, resolution, origin=(0.0, 0.0)):
        self.occ = np.asarray(occ, dtype=bool)
        self.resolution_m = float(resolution)
        self.origin = np.asarray(origin, dtype=float)

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
        #true if the straight line from p to q crosses only free cells
        length = np.hypot(q[0] - p[0], q[1] - p[1])
        n = max(2, int(np.ceil(length / (0.5 * self.resolution_m))) + 1)
        return all(self.is_free(p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1]))
                   for s in np.linspace(0, 1, n))

    def inflate(self, radius):
        #grow every obstacle by `radius` metres. Planning for a point on the
        #inflated map keeps a robot of that radius clear of real obstacles
        n = int(np.ceil(radius / self.resolution_m))
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
        disk = xx ** 2 + yy ** 2 <= n * n
        return OccupancyGrid(binary_dilation(self.occ, structure=disk), self.resolution_m, self.origin)
    @property
    def extent(self):
        #(x_min, x_max, y_min, y_max), for plotting with matplotlib
        h, w = self.occ.shape
        return (self.origin[0], self.origin[0] + w * self.resolution_m,
                self.origin[1], self.origin[1] + h * self.resolution_m)


def build_ward_grid(resolution=0.05):
    #the planner's map, made from the same boxes as the MuJoCo scene
    #later: replace with Person 1's map built from the LiDAR
    occ = np.zeros((int(WARD_HEIGHT / resolution), int(WARD_WIDTH / resolution)), bool)
    for _, x0, x1, y0, y1, _ in WARD_BOXES:
        occ[int(y0 / resolution):int(np.ceil(y1 / resolution)),
            int(x0 / resolution):int(np.ceil(x1 / resolution))] = True
    return OccupancyGrid(occ, resolution)


def astar(grid, start_xy, goal_xy):
    #A* search: the shortest path through free cells, moving in 8 directions
    #it always expands the cell with the lowest (distance so far + straight-line
    #distance still to go), so it heads for the goal without missing shortcuts
    start, goal = grid.world_to_cell(*start_xy), grid.world_to_cell(*goal_xy)
    for label, cell in (("start", start), ("goal", goal)):
        if not grid.in_bounds(*cell) or grid.occ[cell]:
            raise ValueError(f"The {label} is outside the map or too close to an obstacle")

    moves = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
             (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    def distance_to_go(c):
        return np.hypot(c[0] - goal[0], c[1] - goal[1])

    queue = [(distance_to_go(start), 0.0, start)]   #(priority, cost so far, cell)
    came_from, cost = {start: None}, {start: 0.0}
    while queue:
        _, cost_so_far, cell = heapq.heappop(queue)
        if cell == goal:
            break
        if cost_so_far > cost[cell]:
            continue
        for dr, dc, step in moves:
            neighbour = (cell[0] + dr, cell[1] + dc)
            if not grid.in_bounds(*neighbour) or grid.occ[neighbour]:
                continue
            if dr and dc and (grid.occ[cell[0] + dr, cell[1]] or grid.occ[cell[0], cell[1] + dc]):
                continue  #don't squeeze between diagonal obstacles
            if cost_so_far + step < cost.get(neighbour, np.inf):
                cost[neighbour], came_from[neighbour] = cost_so_far + step, cell
                heapq.heappush(queue, (cost_so_far + step + distance_to_go(neighbour), cost_so_far + step, neighbour))
    else:
        raise RuntimeError("A* found no path")
    cells = []  #walk back from the goal to the start
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
    #round off corners: replace each corner with two points 1/4 of the way
    #along its sides, and repeat
    p = np.asarray(points, dtype=float)
    for _ in range(iterations):
        q = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        q.append(p[-1])
        p = np.array(q)
    return p


def plan_to_bedside(grid, start_xy, park_pose, approach_distance=0.8):
    #plan start -> dock, ending with a straight, head-on approach
    #
    #head-on matters: a stopped robot can't correct a sideways error
    #(controllability drops to rank 2), so sideways error must be gone
    #before the robot slows down at the bedside
    park_x, park_y, park_theta = park_pose
    approach_xy = np.array([park_x - approach_distance * np.cos(park_theta),
                            park_y - approach_distance * np.sin(park_theta)])
    planning_grid = grid.inflate(ROBOT_RADIUS + SAFETY_MARGIN)
    raw_astar_path = astar(planning_grid, start_xy, approach_xy)
    smooth_path = chaikin(np.vstack([shortcut(planning_grid, raw_astar_path), [[park_x, park_y]]]))

    body_grid = grid.inflate(ROBOT_RADIUS) #the smoothed path must still be clear
    for a, b in zip(smooth_path[:-1], smooth_path[1:]):
        if not body_grid.segment_free(a, b):
            raise RuntimeError("Smoothed path clips an obstacle; increase SAFETY_MARGIN")
    return smooth_path