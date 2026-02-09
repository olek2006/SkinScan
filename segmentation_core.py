import cv2
import numpy as np

COIN_DIAMETER_MM = 23.5


def remove_hair(image_bgr: np.ndarray):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(
        hair_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )
    inpainted = cv2.inpaint(image_bgr, hair_mask, 3, cv2.INPAINT_TELEA)
    return inpainted, hair_mask


def kmeans_lesion_mask(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]

    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab = cv2.merge((l_eq, a, b))

    lab_reshaped = lab.reshape(-1, 3).astype(np.float32)

    _, labels, centers = cv2.kmeans(
        lab_reshaped,
        3,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        5,
        cv2.KMEANS_PP_CENTERS
    )

    labels = labels.reshape(h, w)
    centers = centers.reshape(3, 3)

    order = np.argsort(centers[:, 0])
    lesion_mask = np.zeros((h, w), dtype=np.uint8)

    for idx in order:
        m = (labels == idx).astype(np.uint8)
        frac = m.mean()
        if 0.01 < frac < 0.8:
            lesion_mask = m
            break

    if lesion_mask.sum() == 0:
        lesion_mask = (labels == order[0]).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    lesion_mask = cv2.dilate(lesion_mask, kernel, iterations=1)

    return lesion_mask * 255


def lesion_score(contour, img_shape):
    H, W = img_shape[:2]

    area = cv2.contourArea(contour)
    if area < 400:
        return -1

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return -1

    x, y, w, h = cv2.boundingRect(contour)

    if x == 0 or y == 0 or x + w >= W or y + h >= H:
        return -1

    aspect = max(w, h) / max(1, min(w, h))
    if aspect > 2.0:
        return -1

    extent = area / (w * h)
    if extent > 0.85:
        return -1

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return -1

    solidity = area / hull_area
    if solidity < 0.9:
        return -1

    circularity = 4 * np.pi * area / (perimeter ** 2)
    if circularity < 0.4:
        return -1

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) < 8:
        return -1

    return circularity + solidity + (1 / aspect)


def clean_lesion_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.copy().astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    scored = []
    for c in contours:
        s = lesion_score(c, mask.shape)
        if s > 0:
            scored.append((s, c))

    if not scored:
        return np.zeros_like(mask)

    _, best = max(scored, key=lambda x: x[0])

    hull = cv2.convexHull(best)
    cleaned = np.zeros_like(mask)
    cv2.drawContours(cleaned, [hull], -1, 255, -1)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.dilate(cleaned, kernel2, iterations=1)

    return cleaned


def detect_coin_ppm(image_bgr: np.ndarray, real_d_mm=COIN_DIAMETER_MM):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    h, w = gray.shape[:2]
    coin_mask = np.zeros((h, w), dtype=np.uint8)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 3,
        param1=100,
        param2=30,
        minRadius=50,
        maxRadius=min(h, w) // 2
    )

    if circles is None:
        return None, None, None

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]

    cv2.circle(coin_mask, (x, y), int(r * 1.15), 255, -1)
    ppm = (2 * r) / real_d_mm

    return ppm, r, coin_mask


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    green = np.zeros_like(image_bgr)
    green[mask > 0] = (0, 255, 0)
    return cv2.addWeighted(image_bgr, 0.7, green, 0.3, 0)


def analyze_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_no_hair, hair_mask = remove_hair(img)
    raw_mask = kmeans_lesion_mask(img_no_hair)

    ppm, _, coin_mask = detect_coin_ppm(img)
    if coin_mask is not None:
        raw_mask[coin_mask > 0] = 0

    final_mask = clean_lesion_mask(raw_mask)

    if coin_mask is not None:
        final_mask[coin_mask > 0] = 0

    area_px = int(np.sum(final_mask > 0))
    area_mm2 = area_px / (ppm ** 2) if ppm else None

    overlay = overlay_mask(img, final_mask)

    return {
        "original": img,
        "no_hair": img_no_hair,
        "hair_mask": hair_mask,
        "raw_mask": raw_mask,
        "final_mask": final_mask,
        "overlay": overlay,
        "area_px": area_px,
        "ppm": ppm,
        "area_mm2": area_mm2,
    }
