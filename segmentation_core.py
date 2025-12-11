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

    # Згладжуємо шум
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)

    # Переходимо в Lab — там канал L добре відокремлює “темне/світле”
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    lab_reshaped = lab.reshape(-1, 3).astype(np.float32)

    # K-means
    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    attempts = 5

    _, labels, centers = cv2.kmeans(lab_reshaped,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

    labels = labels.reshape(h, w)
    centers = centers.reshape(K, 3)


    L_values = centers[:, 0]
    order = np.argsort(L_values)

    lesion_mask = np.zeros((h, w), dtype=np.uint8)

    for idx in order:
        cluster_mask = (labels == idx).astype(np.uint8)
        frac = cluster_mask.mean()


        if 0.01 < frac < 0.8:
            lesion_mask = cluster_mask
            break

    if lesion_mask.sum() == 0:
        lesion_mask = (labels == order[0]).astype(np.uint8)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    lesion_mask = cv2.dilate(lesion_mask, kernel, iterations=1)

    return lesion_mask * 255


def clean_lesion_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.copy().astype(np.uint8)


    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask


    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 300:
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = max(w, h) / max(1.0, min(w, h))


        if aspect_ratio > 6 and area < 5000:
            continue

        valid.append(c)

    if not valid:
        return np.zeros_like(mask)


    main = max(valid, key=cv2.contourArea)


    hull = cv2.convexHull(main)

    cleaned = np.zeros_like(mask)
    cv2.drawContours(cleaned, [hull], -1, 255, thickness=-1)


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

    # Беремо трохи БІЛЬШЕ ніж радіус — щоб точно виключити всю монету
    cv2.circle(coin_mask, (x, y), int(r * 1.15), 255, -1)

    ppm = (2 * r) / real_d_mm
    return ppm, r, coin_mask


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:

    overlay = image_bgr.copy()
    green = np.zeros_like(image_bgr)
    green[mask > 0] = (0, 255, 0)

    blended = cv2.addWeighted(image_bgr, 0.7, green, 0.3, 0)
    return blended


def analyze_image(image_path: str):
    print(f"\n--- АНАЛІЗ ЗОБРАЖЕННЯ: {image_path} ---")

    img = cv2.imread(image_path)
    if img is None:
        print("❌ Неможливо відкрити зображення.")
        return None


    img_no_hair, hair_mask = remove_hair(img)
    print("✅ Волосся оброблено (inpaint).")


    raw_mask = kmeans_lesion_mask(img_no_hair)
    print("✅ Початкова маска утворення отримана (k-means).")


    raw_mask = kmeans_lesion_mask(img_no_hair)
    print("✅ Початкова маска утворення отримана (k-means).")


    ppm, radius_px, coin_mask = detect_coin_ppm(img)


    if coin_mask is not None:
        raw_mask[coin_mask > 0] = 0
        print("✅ Монета видалена з raw_mask")


    final_mask = clean_lesion_mask(raw_mask)
    print("✅ Маска очищена й згладжена по краю (convex hull).")


    if coin_mask is not None:
        final_mask[coin_mask > 0] = 0
        print("✅ Зона монети виключена з маски утворення")


    area_px = int(np.sum(final_mask > 0))
    print(f"📏 Площа утворення (пікселі): {area_px}")


    if ppm is not None:
        area_mm2 = area_px / (ppm ** 2)
        print(f"✨ Площа утворення: {area_mm2:.2f} мм²")
    else:
        area_mm2 = None
        print("✨ Площу в мм² не вдалося обчислити (монета не знайдена).")


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


