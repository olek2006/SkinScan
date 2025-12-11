import cv2
import numpy as np

def asymmetry_score(mask: np.ndarray) -> float:
    mask = (mask > 0).astype(np.uint8)
    area = mask.sum()
    if area == 0:
        return 0.0

    ys, xs = np.where(mask == 1)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    crop = mask[y0:y1+1, x0:x1+1]

    if crop.size == 0:
        return 0.0

    if crop.shape[0] % 2 != 0:
        crop = crop[:-1, :]
    if crop.shape[1] % 2 != 0:
        crop = crop[:, :-1]

    flip_h = np.fliplr(crop)
    flip_v = np.flipud(crop)

    diff = (np.logical_xor(crop, flip_h).sum() +
            np.logical_xor(crop, flip_v).sum())

    return float(np.clip(diff / (2 * area), 0.0, 1.0))


def border_irregularity(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    c = max(contours, key=cv2.contourArea)
    A = cv2.contourArea(c)
    P = cv2.arcLength(c, True)
    if P == 0:
        return 0.0

    return max(0.0, 1 - (4 * np.pi * A) / (P ** 2))


def color_variation(image, mask):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pixels = lab[mask > 0]
    if len(pixels) == 0:
        return 0.0
    return float(np.std(pixels[:, 1]) + np.std(pixels[:, 2]))


def equivalent_diameter(area_mm2):
    return float(2 * np.sqrt(area_mm2 / np.pi))


def abcd_analysis(image, mask, area_mm2):
    A = asymmetry_score(mask)
    B = border_irregularity(mask)
    C = color_variation(image, mask)
    D = equivalent_diameter(area_mm2)

    risk = (
        2.0 * A +
        1.8 * B +
        1.2 * (C / 50) +
        2.5 * (D > 6)
    )

    return {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "risk_abcd": risk
    }
