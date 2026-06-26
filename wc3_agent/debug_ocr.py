"""Отладка OCR — сохраняет захваченные регионы и пробует разные методы чтения."""
import sys, json, cv2, numpy as np, pytesseract
sys.path.insert(0, ".")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from env.screen import ScreenCapture
from env.reward import _ocr_number, _ocr_food
from config import RESOURCE_REGIONS_FILE

sc = ScreenCapture()
sc.update_window_pos()

with open(RESOURCE_REGIONS_FILE) as f:
    regions = json.load(f)

for name, r in regions.items():
    # Захватываем регион
    raw = sc._sct.grab({
        "left":   sc._win_x + r["x"],
        "top":    sc._win_y + r["y"],
        "width":  r["w"],
        "height": r["h"],
    })
    img = np.array(raw, dtype=np.uint8)[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Сохраняем оригинал
    cv2.imwrite(f"debug_{name}_orig.png", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Пробуем разные пороги
    for thresh_val in [40, 60, 80, 100, 128, 150]:
        big = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(big, thresh_val, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh,
            config="--psm 7 -c tessedit_char_whitelist=0123456789/").strip()
        print(f"[{name}] thresh={thresh_val}: '{text}'")

    # CLAHE + Otsu (новый метод)
    h, w2 = img.shape[:2]
    crop = img[:, :int(w2 * 0.65)]
    gray2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    big2 = cv2.resize(gray2, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    enhanced = clahe.apply(big2)
    _, thresh_clahe = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"debug_{name}_clahe.png", thresh_clahe)
    text = pytesseract.image_to_string(thresh_clahe,
        config="--psm 7 -c tessedit_char_whitelist=0123456789/").strip()
    print(f"[{name}] CLAHE+Otsu: '{text}'")

    # Итог, который реально использует агент: голосование по нескольким порогам
    voted = _ocr_food(img) if name == "food" else _ocr_number(img)
    print(f"[{name}] ИТОГ (голосование) = {voted}")
    print()

print("Изображения сохранены: debug_gold_orig.png, debug_lumber_orig.png, debug_food_orig.png")
print("Открой их и посмотри что захвачено.")
