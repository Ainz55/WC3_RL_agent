"""
Reward computation.

Терминальные награды (победа/поражение) — template matching.
Промежуточные награды — OCR цифр ресурсов (золото, дерево, еда).

Формула за шаг:
  reward = REWARD_STEP
         + REWARD_PER_GOLD   * (gold_now - gold_prev)      если выросло
         + REWARD_PER_LUMBER * (lumber_now - lumber_prev)   если выросло
         + REWARD_PER_FOOD   * (food_now - food_prev)       если выросло (юниты обучены)
         + REWARD_FOOD_LOST  * (food_prev - food_now)       если упало (юниты погибли)
"""
import json
import re
import threading
from collections import Counter
import cv2
import numpy as np
import pytesseract
from pathlib import Path
from env.screen import player_mask
from config import (
    TEMPLATES_DIR, RESOURCE_REGIONS_FILE, TESSERACT_CMD, OCR_THRESHOLDS,
    REWARD_WIN, REWARD_LOSE, REWARD_STEP,
    REWARD_PER_GOLD, REWARD_PER_LUMBER, REWARD_PER_FOOD, REWARD_FOOD_LOST,
    REWARD_NEW_BUILDING, REWARD_NEW_FARM, REWARD_ARMY_PEAK,
    REWARD_ENEMY_KILL, MAX_ENEMY_DELTA,
    MAX_GOLD_DELTA, MAX_LUMBER_DELTA, MAX_FOOD_DELTA,
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

MATCH_THRESHOLD = 0.62   # порог совпадения шаблона победы/поражения (снижен с 0.75:
                         # захват темнее эталона, при 0.75 поражение часто не ловилось)
_OCR_CFG = "--psm 7 -c tessedit_char_whitelist=0123456789/"


# ── Template loading ─────────────────────────────────────────────────────────

def _load_template(name: str):
    path = TEMPLATES_DIR / f"{name}.png"
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


# ── OCR helpers ──────────────────────────────────────────────────────────────

def _ocr_candidates(img_bgr: np.ndarray) -> list[str]:
    """
    Распознаёт текст региона при НЕСКОЛЬКИХ порогах бинаризации.

    У золота/дерева/еды разная яркость, поэтому один фиксированный порог не
    читает все три (проверено: при 60 дерево пустое, при 128 золото = '300').
    Возвращаем непустые результаты для каждого порога — выше по стеку из них
    голосованием выбирается итог.
    """
    h, w = img_bgr.shape[:2]
    # Широкий регион еды (120px) содержит справа "Нет ..." — обрезаем.
    # Узкие регионы золота/дерева (~80px) не трогаем.
    if w > 100:
        img_bgr = img_bgr[:, :int(w * 0.65)]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    texts = []
    for t in OCR_THRESHOLDS:
        _, thresh = cv2.threshold(big, t, 255, cv2.THRESH_BINARY)
        txt = pytesseract.image_to_string(thresh, config=_OCR_CFG).strip()
        if txt:
            texts.append(txt)
    return texts


def _vote(values: list):
    """Самое частое значение (отсекает единичные ошибки распознавания)."""
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _ocr_number(img_bgr: np.ndarray) -> int | None:
    """Одно целое число (золото/дерево) — голосование по нескольким порогам."""
    cands = []
    for txt in _ocr_candidates(img_bgr):
        nums = re.findall(r'\d+', txt)
        if nums:
            cands.append(int(nums[0]))
    return _vote(cands)


def _ocr_food(img_bgr: np.ndarray) -> tuple[int | None, int | None]:
    """
    Еда в формате current/max → (current, max), голосование по порогам.
    Приоритет у результатов, где распознаны ОБА числа (пара current/max).
    """
    pairs, curs = [], []
    for txt in _ocr_candidates(img_bgr):
        nums = re.findall(r'\d+', txt)
        if len(nums) >= 2:
            pairs.append((int(nums[0]), int(nums[1])))
            curs.append(int(nums[0]))
        elif len(nums) == 1:
            curs.append(int(nums[0]))
    if pairs:
        return _vote(pairs)
    if curs:
        return _vote(curs), None
    return None, None


# ── Resource reader ──────────────────────────────────────────────────────────

class ResourceReader:
    """
    Читает золото, дерево и еду в фоновом потоке.
    Основной поток берёт последнее готовое значение без ожидания.
    """

    def __init__(self, screen_capture):
        self._sc = screen_capture
        self._regions = self._load_regions()
        self._cache: dict[str, int | None] = {"gold": None, "lumber": None, "food": None}
        self._lock = threading.Lock()
        self._busy = False   # идёт ли сейчас OCR

    def _load_regions(self) -> dict | None:
        if not RESOURCE_REGIONS_FILE.exists():
            print("[ResourceReader] resource_regions.json не найден.")
            print("  Запусти: python calibrate.py resources")
            return None
        with open(RESOURCE_REGIONS_FILE) as f:
            return json.load(f)

    def _grab_region(self, r: dict) -> np.ndarray:
        raw = self._sc._sct.grab({
            "left":   self._sc._win_x + r["x"],
            "top":    self._sc._win_y + r["y"],
            "width":  r["w"],
            "height": r["h"],
        })
        img = np.array(raw, dtype=np.uint8)[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _run_ocr(self, images: dict):
        """OCR в фоновом потоке — получает уже захваченные изображения."""
        food_cur, food_max = _ocr_food(images["food"])
        result = {
            "gold":     _ocr_number(images["gold"]),
            "lumber":   _ocr_number(images["lumber"]),
            "food_cur": food_cur,   # текущие юниты (5 из 5/12)
            "food_max": food_max,   # максимум еды / фермы (12 из 5/12)
        }
        with self._lock:
            self._cache = result
            self._busy = False

    def request(self):
        """
        Скриншот делаем здесь (главный поток — mss безопасен),
        OCR запускаем в фоне.
        """
        if self._busy or self._regions is None:
            return
        # Захватываем изображения в главном потоке
        images = {
            "gold":   self._grab_region(self._regions["gold"]),
            "lumber": self._grab_region(self._regions["lumber"]),
            "food":   self._grab_region(self._regions["food"]),
        }
        self._busy = True
        threading.Thread(target=self._run_ocr, args=(images,), daemon=True).start()

    def read(self) -> dict[str, int | None]:
        """Возвращает {'gold', 'lumber', 'food_cur', 'food_max'}."""
        with self._lock:
            return dict(self._cache)


# ── Building detector ────────────────────────────────────────────────────────

class BuildingDetector:
    """
    Определяет появление новых зданий сравнивая снимки миникарты.

    Принцип:
      - Здания = неподвижные цветные пятна на миникарте.
      - Юниты тоже цветные, но мелкие (2-4 пикселя) и постоянно двигаются.
      - После постройки здания на миникарте появляется НОВЫЙ крупный
        цветной кластер (8+ пикселей) в районе базы игрока.
      - Сравниваем снимок ДО постройки (reset/последнее обновление)
        и ПОСЛЕ (cooldown истёк) → находим новые крупные кластеры.
    """

    _MIN_PIXELS = 8      # минимальный размер кластера = здание (не юнит)

    def __init__(self, screen_capture):
        self._sc = screen_capture
        self._prev_mask: np.ndarray | None = None

    @staticmethod
    def _red_mask(mm_bgr: np.ndarray) -> np.ndarray:
        """Маска пикселей цвета игрока (диапазон HSV из config)."""
        return player_mask(mm_bgr)

    def reset(self):
        """Сохраняет базовый снимок миникарты в начале эпизода."""
        self._prev_mask = self._red_mask(self._sc.grab_minimap())

    def check(self, base_cell: int = 4) -> int:
        """
        Ищет новые здания в зоне базы (±1 ячейка вокруг base_cell).
        Возвращает количество новых зданий (0 если ничего нет).

        Проверяет только зону базы → вражеские здания не засчитываются.
        """
        if self._prev_mask is None:
            return 0

        mm = self._sc.grab_minimap()
        h, w = mm.shape[:2]

        # Ограничиваем поиск зоной базы ±1 ячейка
        row, col = base_cell // 3, base_cell % 3
        ch, cw = h // 3, w // 3
        y1 = max(0, (row - 1) * ch)
        y2 = min(h, (row + 2) * ch)
        x1 = max(0, (col - 1) * cw)
        x2 = min(w, (col + 2) * cw)

        curr_full = self._red_mask(mm)
        curr_zone = curr_full[y1:y2, x1:x2]
        prev_zone = self._prev_mask[y1:y2, x1:x2]

        # Новые пиксели = те что появились с последнего снимка
        new_px = cv2.bitwise_and(curr_zone, cv2.bitwise_not(prev_zone))

        # Убираем шум и одиночные пиксели-юниты — оставляем здания
        kernel = np.ones((3, 3), np.uint8)
        new_px = cv2.morphologyEx(new_px, cv2.MORPH_OPEN, kernel)

        # Считаем отдельные крупные кластеры
        n, _, stats, _ = cv2.connectedComponentsWithStats(new_px)
        new_buildings = sum(
            1 for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= self._MIN_PIXELS
        )

        if new_buildings > 0:
            # Обновляем базовый снимок чтобы следующая проверка видела уже с этим зданием
            self._prev_mask = curr_full
            print(f"  [BuildingDetector] +{new_buildings} зданий в зоне базы "
                  f"(ячейка {base_cell})", flush=True)

        return new_buildings


# ── Main detector ─────────────────────────────────────────────────────────────

class RewardDetector:
    # Сколько раз подряд шаблон должен совпасть прежде чем признать win/lose.
    # Защита от ложных срабатываний во время анимации или боевых эффектов.
    _CONFIRM_STREAK = 2

    def __init__(self, screen_capture=None):
        self._victory_tmpl     = _load_template("victory")
        self._defeat_tmpl      = _load_template("defeat")
        self._res_reader       = ResourceReader(screen_capture) if screen_capture else None
        self.building_detector = BuildingDetector(screen_capture) if screen_capture else None
        self._prev: dict[str, int | None] = {
            "gold": None, "lumber": None, "food_cur": None, "food_max": None
        }
        # Максимум размера армии (supply) за эпизод — для награды «стоимость армии»
        self._food_peak: int | None = None
        # Кол-во красных (враждебных) пикселей на миникарте — для награды за убийства
        self._hostile_prev: int | None = None
        # Для подтверждения win/lose несколькими последовательными проверками
        self._result_candidate: str | None = None
        self._result_streak: int = 0

    def _match(self, scene_bgr: np.ndarray, tmpl_gray) -> float:
        if tmpl_gray is None:
            return 0.0
        scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
        th, tw = tmpl_gray.shape[:2]
        sh, sw = scene_gray.shape[:2]
        if th > sh or tw > sw:
            return 0.0
        result = cv2.matchTemplate(scene_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
        return float(result.max())

    def detect_result(self, center_bgr: np.ndarray) -> str | None:
        """
        Возвращает 'win'/'lose' только после _CONFIRM_STREAK подряд совпадений.
        Это устраняет ложные срабатывания во время анимации и боевых эффектов.
        """
        candidate = None
        if self._match(center_bgr, self._victory_tmpl) >= MATCH_THRESHOLD:
            candidate = "win"
        elif self._match(center_bgr, self._defeat_tmpl) >= MATCH_THRESHOLD:
            candidate = "lose"

        if candidate == self._result_candidate and candidate is not None:
            self._result_streak += 1
        else:
            self._result_candidate = candidate
            self._result_streak = 1

        if candidate is not None and self._result_streak >= self._CONFIRM_STREAK:
            return candidate
        return None

    def compute_reward(
        self,
        center_bgr: np.ndarray,
    ) -> tuple[float, bool, str | None]:

        # Проверяем конец игры (center_bgr=None если пропускаем этот шаг)
        if center_bgr is not None:
            result = self.detect_result(center_bgr)
            if result == "win":
                return REWARD_WIN, True, "win"
            if result == "lose":
                return REWARD_LOSE, True, "lose"

        # OCR запускается в фоне — основной цикл не ждёт
        reward = REWARD_STEP
        if self._res_reader is not None:
            self._res_reader.request()     # пнуть фоновый поток
            cur = self._res_reader.read()  # взять последнее готовое значение
            prev = self._prev
            new_prev = dict(prev)

            def shaped(key, max_delta, pos_w, neg_w=0.0) -> float:
                """
                Награда за изменение ресурса с отсевом ошибок OCR.
                Скачок больше max_delta за один шаг считаем мусором: не начисляем
                награду И не обновляем prev (чтобы сравнивать со старым значением).
                """
                v, p = cur.get(key), prev.get(key)
                if v is None:
                    return 0.0
                if p is None:
                    new_prev[key] = v
                    return 0.0
                d = v - p
                if abs(d) > max_delta:
                    return 0.0
                new_prev[key] = v
                if d > 0:
                    return pos_w * d
                if d < 0:
                    return neg_w * abs(d)
                return 0.0

            reward += shaped("gold",     MAX_GOLD_DELTA,   REWARD_PER_GOLD)
            reward += shaped("lumber",   MAX_LUMBER_DELTA, REWARD_PER_LUMBER)
            # Юниты: обучен (+REWARD_PER_FOOD) или погиб (REWARD_FOOD_LOST)
            reward += shaped("food_cur", MAX_FOOD_DELTA,   REWARD_PER_FOOD, REWARD_FOOD_LOST)
            # Ферма/ратуша построена → вырос максимум еды
            reward += shaped("food_max", MAX_FOOD_DELTA,   REWARD_NEW_FARM)

            self._prev = new_prev

            # «Стоимость армии»: награда за каждый новый максимум supply.
            # Первый замер задаёт точку отсчёта без награды (не платим за стартовых рабочих).
            fc = new_prev.get("food_cur")
            if fc is not None:
                if self._food_peak is None:
                    self._food_peak = fc
                elif fc > self._food_peak:
                    reward += REWARD_ARMY_PEAK * (fc - self._food_peak)
                    self._food_peak = fc

        return reward, False, None

    def get_resources(self) -> dict[str, int | None]:
        """Последние достоверные значения ресурсов (для вектора stats наблюдения)."""
        return dict(self._prev)

    def check_enemy_kills(self, minimap_bgr) -> float:
        """
        Прокси «убийства врагов/крипов»: красное на миникарте = враждебные
        (игрок бирюзовый). Награда за УМЕНЬШЕНИЕ красного с прошлой проверки.
        Шумный сигнал → малый вес + кламп больших скачков (туман/прокрутка).
        """
        red = cv2.inRange(minimap_bgr,
                          np.array([0, 0, 150], dtype=np.uint8),
                          np.array([80, 80, 255], dtype=np.uint8))
        cur = int((red > 0).sum())
        reward = 0.0
        if self._hostile_prev is not None:
            drop = self._hostile_prev - cur          # уменьшение = кого-то убили/ушли
            if drop > 0:
                reward = REWARD_ENEMY_KILL * min(drop, MAX_ENEMY_DELTA)
        self._hostile_prev = cur
        return reward

    def reset(self):
        self._prev = {"gold": None, "lumber": None, "food_cur": None, "food_max": None}
        self._food_peak = None
        self._hostile_prev = None
        self._result_candidate = None
        self._result_streak = 0
        if self.building_detector is not None:
            self.building_detector.reset()
