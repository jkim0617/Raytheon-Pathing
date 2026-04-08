import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_disk_kernel(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.array([[1]], dtype=np.uint8)
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = (x * x + y * y) <= radius * radius
    return mask.astype(np.uint8)


def remove_small_components(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    h, w = binary_mask.shape
    flood = binary_mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary_mask, flood_inv)


def capture_image(camera_index: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame from camera.")
    return frame


def load_image(image_file: str) -> np.ndarray:
    img = cv2.imread(image_file)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_file}")
    return img


def crop_roi(img_bgr: np.ndarray, use_roi: bool, fixed_roi: tuple[int, int, int, int]) -> np.ndarray:
    if not use_roi:
        return img_bgr.copy()
    x, y, w, h = fixed_roi
    return img_bgr[y:y + h, x:x + w].copy()


def image_to_occ_grid(
    field_bgr: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    field_hue_min: float,
    field_hue_max: float,
    field_sat_min: float,
    field_val_min: float,
    min_obstacle_area: int,
    open_radius: int,
    close_radius: int,
    occupancy_threshold: float,
    inflate_radius: int,
):
    hsv = cv2.cvtColor(field_bgr, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    hue_min_cv = int(field_hue_min * 179)
    hue_max_cv = int(field_hue_max * 179)
    sat_min_cv = int(field_sat_min * 255)
    val_min_cv = int(field_val_min * 255)

    field_mask = (
        (h > hue_min_cv) &
        (h < hue_max_cv) &
        (s > sat_min_cv) &
        (v > val_min_cv)
    ).astype(np.uint8) * 255

    obstacle_mask = cv2.bitwise_not(field_mask)
    obstacle_mask = remove_small_components(obstacle_mask, min_obstacle_area)

    open_kernel = make_disk_kernel(open_radius)
    close_kernel = make_disk_kernel(close_radius)

    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, open_kernel)
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, close_kernel)
    obstacle_mask = fill_holes(obstacle_mask)

    img_h, img_w = obstacle_mask.shape
    cell_h = img_h // grid_rows
    cell_w = img_w // grid_cols

    occ_grid = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    for r in range(grid_rows):
        for c in range(grid_cols):
            r1 = r * cell_h
            r2 = img_h if r == grid_rows - 1 else (r + 1) * cell_h
            c1 = c * cell_w
            c2 = img_w if c == grid_cols - 1 else (c + 1) * cell_w

            cell_mask = obstacle_mask[r1:r2, c1:c2]
            if np.mean(cell_mask > 0) > occupancy_threshold:
                occ_grid[r, c] = 1

    if inflate_radius > 0:
        inflate_kernel = make_disk_kernel(inflate_radius)
        occ_grid = cv2.dilate(occ_grid, inflate_kernel)

    return field_mask, obstacle_mask, occ_grid


def save_occ_grid_csv(occ_grid: np.ndarray, filename: str = "occGrid.csv") -> None:
    np.savetxt(filename, occ_grid, fmt="%d", delimiter=",")


def show_vision_results(field_bgr: np.ndarray, obstacle_mask: np.ndarray, occ_grid: np.ndarray) -> None:
    field_rgb = cv2.cvtColor(field_bgr, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(field_rgb)
    plt.title("Field ROI")
    plt.axis("off")

    plt.figure()
    plt.imshow(obstacle_mask, cmap="gray")
    plt.title("Obstacle Mask")
    plt.axis("off")

    plt.figure()
    plt.imshow(occ_grid, cmap="gray", origin="upper")
    plt.title("Occupancy Grid")
    plt.axis("equal")

    plt.show()


def overlay_path_on_image(field_bgr: np.ndarray, occ_grid: np.ndarray, best_path: np.ndarray) -> None:
    img_rgb = cv2.cvtColor(field_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = field_bgr.shape[:2]
    grid_rows, grid_cols = occ_grid.shape

    path_px = []
    for xg, yg in best_path:
        xp = (xg - 0.5) / grid_cols * img_w
        yp = img_h - (yg - 0.5) / grid_rows * img_h
        path_px.append((xp, yp))
    path_px = np.array(path_px)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img_rgb)

    row_occ, col_occ = np.where(occ_grid == 1)
    cell_w = img_w / grid_cols
    cell_h = img_h / grid_rows

    for r, c in zip(row_occ, col_occ):
        x1 = c * cell_w
        y1 = img_h - (r + 1) * cell_h
        rect = plt.Rectangle((x1, y1), cell_w, cell_h, fill=False, edgecolor="red", linewidth=0.5)
        ax.add_patch(rect)

    ax.plot(path_px[:, 0], path_px[:, 1], "-b", linewidth=3)
    ax.plot(path_px[:, 0], path_px[:, 1], "yo", markersize=5)
    ax.plot(path_px[0, 0], path_px[0, 1], "gs", markersize=10)
    ax.plot(path_px[-1, 0], path_px[-1, 1], "ms", markersize=10)
    ax.set_title("Best Path Over Original Image With Obstacle Cells")
    plt.show()