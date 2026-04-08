import csv
import math
import numpy as np


def path_to_commands(path: np.ndarray):
    commands = []

    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]

        dx = x2 - x1
        dy = y2 - y1

        distance = math.hypot(dx, dy)

        # North=0, East=90, South=180, West=270
        angle = math.degrees(math.atan2(dx, dy))
        if angle < 0:
            angle += 360

        commands.append({
            "step": i,
            "start_x": float(x1),
            "start_y": float(y1),
            "end_x": float(x2),
            "end_y": float(y2),
            "distance": float(distance),
            "angle": float(angle),
        })

    return commands


def save_commands_csv(commands, filename: str = "commands.csv") -> None:
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "start_x", "start_y", "end_x", "end_y", "distance", "angle"])
        for cmd in commands:
            writer.writerow([
                cmd["step"],
                cmd["start_x"],
                cmd["start_y"],
                cmd["end_x"],
                cmd["end_y"],
                cmd["distance"],
                cmd["angle"],
            ])


def load_path_csv(filename: str) -> np.ndarray:
    return np.loadtxt(filename, delimiter=",")