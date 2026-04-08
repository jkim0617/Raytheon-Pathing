import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing


def raster_line_xy(p1, p2, grid_rows, grid_cols):
    x1, y1 = p1
    x2, y2 = p2

    n = int(max(abs(round(x2 - x1)), abs(round(y2 - y1))) + 1)
    xs = np.round(np.linspace(x1, x2, n)).astype(int)
    ys = np.round(np.linspace(y1, y2, n)).astype(int)

    xs = np.clip(xs, 1, grid_cols)
    ys = np.clip(ys, 1, grid_rows)

    pts = np.column_stack((xs, ys))
    pts = np.unique(pts, axis=0)
    return pts


def path_cost_grid(x, start_pnt, end_pnt, occ_grid):
    grid_rows, grid_cols = occ_grid.shape

    waypoints = np.array(x, dtype=float).reshape(-1, 2)
    waypoints[:, 0] = np.clip(waypoints[:, 0], 1, grid_cols)
    waypoints[:, 1] = np.clip(waypoints[:, 1], 1, grid_rows)

    path = np.vstack([start_pnt, waypoints, end_pnt])

    diffs = np.diff(path, axis=0)
    seg_len = np.sqrt(np.sum(diffs ** 2, axis=1))
    path_len = np.sum(seg_len)

    smooth_pen = 0.0
    for i in range(1, len(path) - 1):
        v1 = path[i] - path[i - 1]
        v2 = path[i + 1] - path[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            cos_angle = np.dot(v1, v2) / (n1 * n2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.acos(cos_angle)
            smooth_pen += angle ** 2

    collision_pen = 0.0
    clearance_pen = 0.0

    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        segment_cells = raster_line_xy(p1, p2, grid_rows, grid_cols)

        for c, y in segment_cells:
            rr = int(np.clip(y, 1, grid_rows)) - 1
            cc = int(np.clip(c, 1, grid_cols)) - 1

            if occ_grid[rr, cc] == 1:
                collision_pen += 1e5
            else:
                r1 = max(0, rr - 1)
                r2 = min(grid_rows - 1, rr + 1)
                c1 = max(0, cc - 1)
                c2 = min(grid_cols - 1, cc + 1)
                if np.any(occ_grid[r1:r2 + 1, c1:c2 + 1]):
                    clearance_pen += 10

    return path_len + 5 * smooth_pen + collision_pen + clearance_pen


def choose_start_goal(
    occ_grid: np.ndarray,
    randomize_start_goal: bool = False,
    start_point: tuple[float, float] = (2, 2),
    goal_mode: str = "topright",
    manual_goal: tuple[float, float] = (23, 23),
    min_random_distance: float | None = None,
):
    grid_rows, grid_cols = occ_grid.shape

    if randomize_start_goal:
        free_rows, free_cols = np.where(occ_grid == 0)
        if len(free_rows) < 2:
            raise RuntimeError("Not enough free cells for start and goal.")

        while True:
            idx = np.random.choice(len(free_rows), size=2, replace=False)
            start = np.array([free_cols[idx[0]] + 1, free_rows[idx[0]] + 1], dtype=float)
            goal = np.array([free_cols[idx[1]] + 1, free_rows[idx[1]] + 1], dtype=float)

            if min_random_distance is None or np.linalg.norm(start - goal) >= min_random_distance:
                return start, goal

    start = np.array(start_point, dtype=float)

    if goal_mode.lower() == "topright":
        goal = np.array([grid_cols - 2, 2], dtype=float)
    else:
        goal = np.array(manual_goal, dtype=float)

    return start, goal


def validate_free_point(point: np.ndarray, occ_grid: np.ndarray, name: str) -> None:
    x, y = point
    rows, cols = occ_grid.shape
    xi = int(round(x))
    yi = int(round(y))

    if not (1 <= xi <= cols and 1 <= yi <= rows):
        raise ValueError(f"{name} is outside the grid.")

    if occ_grid[yi - 1, xi - 1] == 1:
        raise ValueError(f"{name} is inside an occupied cell.")


def run_annealing(
    occ_grid: np.ndarray,
    start_pnt: np.ndarray,
    end_pnt: np.ndarray,
    min_waypoints: int = 1,
    max_waypoints: int = 8,
    improvement_threshold: float = 0.01,
    required_stable: int = 2,
    maxiter: int = 500,
):
    grid_rows, grid_cols = occ_grid.shape

    prev_cost = float("inf")
    best_cost = float("inf")
    best_path = None
    best_num_waypoints = 0
    stable_count = 0

    for num_waypoints in range(min_waypoints, max_waypoints + 1):
        print(f"\nTrying {num_waypoints} waypoints")

        bounds = []
        x0 = []

        for i in range(num_waypoints):
            bounds.append((1, grid_cols))
            bounds.append((1, grid_rows))

            alpha = (i + 1) / (num_waypoints + 1)
            pt = (1 - alpha) * start_pnt + alpha * end_pnt
            x0.extend([pt[0], pt[1]])

        result = dual_annealing(
            lambda x: path_cost_grid(x, start_pnt, end_pnt, occ_grid),
            bounds=bounds,
            x0=np.array(x0, dtype=float),
            maxiter=maxiter,
            no_local_search=True,
        )

        x_best = result.x
        f_best = result.fun
        path = np.vstack([start_pnt, np.array(x_best).reshape(-1, 2), end_pnt])

        print(f"Cost = {f_best:.4f}")

        if f_best < best_cost:
            best_cost = f_best
            best_path = path
            best_num_waypoints = num_waypoints

        if math.isfinite(prev_cost):
            improvement = (prev_cost - f_best) / prev_cost
            print(f"Improvement = {improvement:.4f}")

            if 0 <= improvement < improvement_threshold:
                stable_count += 1
                print(f"Small improvement ({stable_count}/{required_stable})")
            else:
                stable_count = 0

            if stable_count >= required_stable:
                print("\nStopping: solution stabilized.")
                break

        prev_cost = f_best

    print(f"\nBest solution used {best_num_waypoints} waypoints")
    print(f"Best cost = {best_cost:.4f}")

    return best_path, best_cost, best_num_waypoints


def plot_occ_grid_with_path(occ_grid: np.ndarray, best_path: np.ndarray) -> None:
    plt.figure(figsize=(7, 7))
    plt.imshow(occ_grid, cmap="gray", origin="upper")
    plt.plot(best_path[:, 0] - 0.5, best_path[:, 1] - 0.5, "-bo", linewidth=2, markersize=4)
    plt.plot(best_path[0, 0] - 0.5, best_path[0, 1] - 0.5, "gs", markersize=10)
    plt.plot(best_path[-1, 0] - 0.5, best_path[-1, 1] - 0.5, "ms", markersize=10)
    plt.title("Best Path on Occupancy Grid")
    plt.axis("equal")
    plt.show()


def save_path_csv(best_path: np.ndarray, full_path_file: str = "full_path.csv", waypoints_file: str = "waypoints.csv") -> None:
    np.savetxt(full_path_file, best_path, fmt="%.6f", delimiter=",")
    waypoints = best_path[1:-1]
    np.savetxt(waypoints_file, waypoints, fmt="%.6f", delimiter=",")