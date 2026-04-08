import numpy as np

from vision import (
    capture_image,
    load_image,
    crop_roi,
    image_to_occ_grid,
    save_occ_grid_csv,
    show_vision_results,
    overlay_path_on_image,
)
from planner import (
    choose_start_goal,
    validate_free_point,
    run_annealing,
    plot_occ_grid_with_path,
    save_path_csv,
)
from commands import path_to_commands, save_commands_csv


def main():
    # =========================
    # USER SETTINGS
    # =========================
    use_camera = False
    camera_index = 0
    image_file = "fieldTest.jpg"

    use_roi = True
    fixed_roi = (50, 50, 900, 500)

    grid_rows = 25
    grid_cols = 25

    field_hue_min = 0.18
    field_hue_max = 0.45
    field_sat_min = 0.20
    field_val_min = 0.15

    min_obstacle_area = 150
    open_radius = 3
    close_radius = 8
    occupancy_threshold = 0.10
    inflate_radius = 1

    randomize_start_goal = True
    start_point = (2, 2)
    goal_mode = "topright"
    manual_goal = (23, 23)
    min_random_distance = None

    min_waypoints = 1
    max_waypoints = 5
    improvement_threshold = 0.01
    required_stable = 2
    maxiter = 200

    show_intermediate_plots = True

    # =========================
    # ACQUIRE IMAGE
    # =========================
    if use_camera:
        img_bgr = capture_image(camera_index)
    else:
        img_bgr = load_image(image_file)

    field_bgr = crop_roi(img_bgr, use_roi, fixed_roi)

    # =========================
    # VISION
    # =========================
    field_mask, obstacle_mask, occ_grid = image_to_occ_grid(
        field_bgr=field_bgr,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        field_hue_min=field_hue_min,
        field_hue_max=field_hue_max,
        field_sat_min=field_sat_min,
        field_val_min=field_val_min,
        min_obstacle_area=min_obstacle_area,
        open_radius=open_radius,
        close_radius=close_radius,
        occupancy_threshold=occupancy_threshold,
        inflate_radius=inflate_radius,
    )

    save_occ_grid_csv(occ_grid, "occGrid.csv")

    # if show_intermediate_plots:
    #     show_vision_results(field_bgr, obstacle_mask, occ_grid)

    # =========================
    # START / GOAL
    # =========================
    start_pnt, end_pnt = choose_start_goal(
        occ_grid=occ_grid,
        randomize_start_goal=randomize_start_goal,
        start_point=start_point,
        goal_mode=goal_mode,
        manual_goal=manual_goal,
        min_random_distance=min_random_distance,
    )

    validate_free_point(start_pnt, occ_grid, "Start")
    validate_free_point(end_pnt, occ_grid, "Goal")

    print("Start:", start_pnt)
    print("Goal:", end_pnt)

    # =========================
    # PLANNER
    # =========================
    best_path, best_cost, best_num_waypoints = run_annealing(
        occ_grid=occ_grid,
        start_pnt=start_pnt,
        end_pnt=end_pnt,
        min_waypoints=min_waypoints,
        max_waypoints=max_waypoints,
        improvement_threshold=improvement_threshold,
        required_stable=required_stable,
        maxiter=maxiter,
    )

    save_path_csv(best_path, "full_path.csv", "waypoints.csv")
    # plot_occ_grid_with_path(occ_grid, best_path)
    overlay_path_on_image(field_bgr, occ_grid, best_path)

    # =========================
    # COMMAND EXPORT
    # =========================
    commands = path_to_commands(best_path)
    save_commands_csv(commands, "commands.csv")

    print("\nSaved:")
    print("  occGrid.csv")
    print("  full_path.csv")
    print("  waypoints.csv")
    print("  commands.csv")
    print(f"\nBest path used {best_num_waypoints} waypoints")
    print(f"Best cost = {best_cost:.4f}")


if __name__ == "__main__":
    main()