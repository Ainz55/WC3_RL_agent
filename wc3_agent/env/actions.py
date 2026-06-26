"""
Action space for WC3 Melee (Human race).

Мышь: PostMessage → WC3 получает клики БЕЗ движения курсора пользователя.
Клавиши: SendInput (pydirectinput) → попадают в активное окно.

Action index:
  0-8   : attack-move на миникарте (3×3 сетка)
  9-17  : камера на миникарте (3×3 сетка)
  18    : выделить всех юнитов (Tab)
  19    : выделить рабочего (backtick `, см. IDLE_WORKER_KEY в config.py)
  20    : обучить крестьянина (P)
  21    : обучить пехотинца (F)
  22    : обучить стрелка (R)
  23    : строить ферму (B→F)
  24    : строить казармы (B→B)
  25    : строить лесопилку (B→L)
  26    : строить Алтарь (B→A)
  27    : строить Сторожевую башню (B→W)
  28    : строить Кузницу (B→S)
  29    : строить Мастерскую (B→V→W)
  30    : строить Магич. лавку (B→V→V)
  31    : строить Тайн. святилище (B→V→R)
  32    : строить Башню грифонов (B→V→G)
  33-39 : продвинутые юниты и герои
  40    : рабочих на золото (Right-click миникарта)
  41-42 : улучшить базу (Крепость/Замок)
  43    : обучить летательную машину (M)
  44    : обучить мортиру (C)
  45    : обучить осадную машину (T)
  46    : ничего не делать
  47    : призвать ополчение (M)
  48    : обучить жреца (P)
  49    : обучить всадника на дракончике (D)
  50    : обучить грифона (G)
  51    : строить Ратушу / восстановить (B→T)
  52    : клавиша W  (вод. элементаль / добыча дерева)
  53    : клавиша V  (Аватар Горн. Короля)
  54    : клавиша E  (Изгнание Кров. мага / Осветит. снаряд)
  55    : клавиша N  (Призыв Феникса)
  56    : клавиша L  (Ускор. добыча дерева)
"""
import time
import json
import math
import ctypes
import cv2
import numpy as np
import win32api
import win32con
import win32gui
import pydirectinput

from env.screen import grab_region_gray, grab_region_bgr
from config import (
    MINIMAP, MINIMAP_GRID, ACTION_DELAY, WINDOW_TITLE, BASE_MINIMAP_CELL,
    GAME_W, GAME_H,
    BUILDINGS_DIR, BUILDING_SEARCH_REGION, BUILDING_MATCH_THRESHOLD,
    RESOURCE_MATCH_THRESHOLD,
    BUILDING_CENTER_ON_BASE, IDLE_WORKER_KEY,
    BUILD_PLACE_DX, BUILD_PLACE_DY,
    TREE_GREEN_HSV_LOW, TREE_GREEN_HSV_HIGH, MIN_TREE_AREA,
    GRASS_HSV_LOW, GRASS_HSV_HIGH,
)

pydirectinput.PAUSE = 0.0

N_ACTIONS = 58

# Текущая ячейка базы — обновляется автоматически в начале каждого эпизода
# через set_base_cell(). Начальное значение берётся из config.
_current_base_cell: int = BASE_MINIMAP_CELL


def set_base_cell(cell: int) -> None:
    """Устанавливает ячейку миникарты для размещения зданий (вызывается из reset)."""
    global _current_base_cell
    _current_base_cell = cell

# ── Найти HWND WC3 ───────────────────────────────────────────────────────────

def get_wc3_hwnd() -> int | None:
    found = []
    def _cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and WINDOW_TITLE in win32gui.GetWindowText(hwnd):
            found.append(hwnd)
    win32gui.EnumWindows(_cb, None)
    return found[0] if found else None


def focus_game_window():
    """Переключает фокус на WC3 через AttachThreadInput."""
    hwnd = get_wc3_hwnd()
    if not hwnd:
        return
    try:
        cur_tid = ctypes.windll.kernel32.GetCurrentThreadId()
        fg_hwnd = win32gui.GetForegroundWindow()
        fg_tid = win32gui.GetWindowThreadProcessId(fg_hwnd)[0] if fg_hwnd else cur_tid
        wc3_tid = win32gui.GetWindowThreadProcessId(hwnd)[0]

        ctypes.windll.user32.AttachThreadInput(cur_tid, fg_tid, True)
        ctypes.windll.user32.AttachThreadInput(fg_tid, wc3_tid, True)
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetForegroundWindow(hwnd)
        ctypes.windll.user32.AttachThreadInput(cur_tid, fg_tid, False)
        ctypes.windll.user32.AttachThreadInput(fg_tid, wc3_tid, False)
        time.sleep(0.15)
    except Exception:
        pass

    fg = win32gui.GetWindowText(win32gui.GetForegroundWindow())
    if WINDOW_TITLE not in fg:
        print(f"[focus] ВНИМАНИЕ: активно '{fg}', а не WC3!")


# ── Ввод мыши через PostMessage (курсор НЕ двигается) ───────────────────────

def _post_click(hwnd: int, client_x: int, client_y: int, button: str = "left"):
    """
    Кликает в точку (client_x, client_y) клиентской области WC3.
    Сохраняет позицию курсора → кликает → возвращает курсор.
    Курсор мигает на доли секунды, но тут же возвращается на место.
    """
    # Конвертируем клиентские координаты в экранные
    screen_x, screen_y = win32gui.ClientToScreen(hwnd, (client_x, client_y))

    # Сохраняем позицию курсора пользователя
    old_pos = win32api.GetCursorPos()

    # Кликаем
    pydirectinput.moveTo(screen_x, screen_y)
    pydirectinput.click(screen_x, screen_y, button=button)

    # Возвращаем курсор на место
    win32api.SetCursorPos(old_pos)


# ── Ввод клавиш через SendInput ───────────────────────────────────────────────

def _key(key: str):
    pydirectinput.keyDown(key)
    time.sleep(0.02)
    pydirectinput.keyUp(key)


def _hotkey(*keys: str):
    for k in keys:
        pydirectinput.keyDown(k)
    time.sleep(0.02)
    for k in reversed(keys):
        pydirectinput.keyUp(k)


# ── Позиции миникарты ────────────────────────────────────────────────────────

def _minimap_client(grid_idx: int) -> tuple[int, int]:
    """Возвращает (x, y) в клиентских координатах WC3 для ячейки миникарты."""
    row = grid_idx // MINIMAP_GRID
    col = grid_idx % MINIMAP_GRID
    cell_w = MINIMAP["w"] // MINIMAP_GRID
    cell_h = MINIMAP["h"] // MINIMAP_GRID
    x = MINIMAP["x"] + col * cell_w + cell_w // 2
    y = MINIMAP["y"] + row * cell_h + cell_h // 2
    return x, y


def _interior_dir() -> tuple[int, int]:
    """Направление от УГЛА базы к открытому центру карты: (sx, sy) ∈ {-1,0,1}."""
    row, col = _current_base_cell // 3, _current_base_cell % 3
    sx = 1 if col == 0 else (-1 if col == 2 else 0)
    sy = 1 if row == 0 else (-1 if row == 2 else 0)
    if sx == 0 and sy == 0:            # база в центре карты — ставим вправо
        sx = 1
    return sx, sy


_build_rot = 0
_last_build_placed = False   # поставилось ли здание в последнем _build (по падению золота)
_last_hero_hired = False     # нанят ли герой в последнем _train (по падению золота)


def last_build_placed() -> bool:
    """True если последняя постройка реально встала (для награды за здание)."""
    return _last_build_placed


def last_hero_hired() -> bool:
    """True если последний найм из Алтаря действительно дал героя (для награды)."""
    return _last_hero_hired


def _open_grass_spots(win_x: int, win_y: int) -> list:
    """
    Точки в ГЛУБИНЕ открытой травы (для размещения зданий), клиентские координаты,
    самые свободные первыми. Открытая трава = крупнейшие зелёные кластеры; «глубокая»
    точка (distance transform) дальше всего от построек/препятствий → свободно.
    """
    bgr = grab_region_bgr(win_x, win_y, BUILDING_SEARCH_REGION)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, np.array(GRASS_HSV_LOW, np.uint8),
                             np.array(GRASS_HSV_HIGH, np.uint8))
    grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(grass)
    order = sorted(range(1, n), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)

    spots = []
    for ci in order[:2]:                       # 2 крупнейшие открытые зоны
        if int(stats[ci, cv2.CC_STAT_AREA]) < 4000:
            break
        cl = (labels == ci).astype(np.uint8) * 255
        bb = 35                                # отступ от края кадра
        cl[:bb, :] = 0; cl[-bb:, :] = 0
        cl[:, :bb] = 0; cl[:, -bb:] = 0
        dist = cv2.distanceTransform(cl, cv2.DIST_L2, 5)
        for _ in range(3):                     # по 3 разнесённых точки из зоны (меньше = быстрее)
            _, mx, _, ml = cv2.minMaxLoc(dist)
            if mx < 28:                         # мало места для здания (зазор)
                break
            spots.append((BUILDING_SEARCH_REGION["x"] + ml[0],
                          BUILDING_SEARCH_REGION["y"] + ml[1]))
            cv2.circle(dist, ml, 85, 0, -1)     # стереть вокруг → следующая точка в стороне
    return spots


def _build(hwnd: int, win_x: int, win_y: int, *keys: str):
    """
    Постройка С ПРОВЕРКОЙ результата (НЕ зависит от Ратуши/ячейки базы):
      1. Камера на базу → выбрать рабочего → ВЕРНУТЬ камеру (backtick её увёл).
      2. B + тип здания → режим размещения.
      3. Ставим в ГЛУБИНУ открытой травы (крупнейшее зелёное поле без построек).
         После клика смотрим, УПАЛО ли золото: упало → поставилось, стоп; ни одна
         точка не подошла → Esc (не долбим по занятому).
    """
    global _last_build_placed
    _center_on_base(hwnd)
    _key(IDLE_WORKER_KEY)              # выбрать рабочего (камера прыгает к нему)
    time.sleep(0.1)
    _center_on_base(hwnd)              # вернуть камеру на базу (рабочий остаётся выбран)
    time.sleep(0.25)                   # дать камере устаканиться

    _key("b")                          # меню постройки
    for k in keys:
        time.sleep(0.04)
        _key(k)
    time.sleep(0.1)                    # ждём курсор размещения

    gold_before = _read_gold(win_x, win_y)
    spots = _open_grass_spots(win_x, win_y)

    placed = False
    tries = 0
    for px, py in spots:
        tries += 1
        _post_click(hwnd, px, py)
        time.sleep(0.4)
        if gold_before is None:        # золото не прочиталось → ставим на первой точке
            placed = True
            break
        ga = _read_gold(win_x, win_y)
        if ga is not None and ga < gold_before - 30:   # золото упало → поставилось
            placed = True
            break

    if not placed:
        _key("escape")                 # свободного места не нашлось — отменить
    _last_build_placed = placed        # для награды за постройку (читает wc3_env)
    print(f"    [build] золото={gold_before} мест-травы={len(spots)} попыток={tries} "
          f"{'ПОСТАВИЛОСЬ' if placed else 'не нашлось места→отмена'}", flush=True)


# ── Выбор здания по экрану (vision) ──────────────────────────────────────────

# Какое здание выделить и какую клавишу нажать для каждого действия найма/апгрейда.
ACTION_BUILD_TRAIN: dict[int, tuple[str, str]] = {
    20: ("townhall", "p"),   # крестьянин
    47: ("townhall", "m"),   # ополчение
    41: ("townhall", "u"),   # апгрейд → крепость
    42: ("townhall", "u"),   # апгрейд → замок
    21: ("barracks", "f"),   # пехотинец
    22: ("barracks", "r"),   # стрелок
    35: ("barracks", "k"),   # рыцарь
    36: ("altar", "p"),      # Паладин
    37: ("altar", "m"),      # Горный Король
    38: ("altar", "a"),      # Верховный маг
    39: ("altar", "b"),      # Чародей Крови
    43: ("workshop", "m"),   # летательная машина
    44: ("workshop", "c"),   # мортира
    45: ("workshop", "t"),   # осадная машина
    33: ("arcane", "s"),     # волшебница
    34: ("arcane", "b"),     # ведьмак
    48: ("arcane", "p"),     # жрец
    49: ("aviary", "d"),     # всадник на дракончике
    50: ("aviary", "g"),     # грифон
}

_building_templates: dict = {}

# У Ратуши три уровня с разным видом (Ратуша→Крепость→Замок) — для логического
# здания "townhall" пробуем все три шаблона, какие захвачены. Остальные здания
# уровней не меняют и используют шаблон со своим именем.
BUILDING_TEMPLATES_FOR: dict[str, list[str]] = {
    "townhall": ["townhall", "keep", "castle"],
}


def _load_building_template(name: str):
    """Лениво грузит шаблон здания (grayscale). Кэширует результат, в т.ч. отсутствие."""
    if name not in _building_templates:
        path = BUILDINGS_DIR / f"{name}.png"
        _building_templates[name] = (
            cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None
        )
    return _building_templates[name]


def _has_any_template(name: str) -> bool:
    """Есть ли хоть один шаблон для логического здания (с учётом уровней Ратуши)."""
    return any(_load_building_template(v) is not None
               for v in BUILDING_TEMPLATES_FOR.get(name, [name]))


def _locate_building(win_x: int, win_y: int, name: str,
                     threshold: float = BUILDING_MATCH_THRESHOLD) -> tuple | None:
    """
    Координаты центра объекта `name` на поле боя по шаблону(ам), в клиентских
    координатах окна. Для "townhall" пробует все уровни. None если не найдено
    (лучший балл ниже threshold).
    """
    variants = [_load_building_template(v)
                for v in BUILDING_TEMPLATES_FOR.get(name, [name])]
    variants = [t for t in variants if t is not None]
    if not variants:
        return None

    scene = grab_region_gray(win_x, win_y, BUILDING_SEARCH_REGION)
    sh, sw = scene.shape[:2]

    best_val, best_xy = -1.0, None
    for tmpl in variants:
        th, tw = tmpl.shape[:2]
        if th > sh or tw > sw:
            continue
        res = cv2.matchTemplate(scene, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_xy = (BUILDING_SEARCH_REGION["x"] + max_loc[0] + tw // 2,
                       BUILDING_SEARCH_REGION["y"] + max_loc[1] + th // 2)

    # `not (>=)` заодно отсекает NaN (бывает на константных областях)
    if best_xy is None or not (best_val >= threshold):
        return None
    return best_xy


def _find_and_click_building(hwnd: int, win_x: int, win_y: int,
                            name: str, button: str = "left") -> bool:
    """Находит объект по шаблону и кликает по нему. True если найдено."""
    xy = _locate_building(win_x, win_y, name)
    if xy is None:
        return False
    _post_click(hwnd, xy[0], xy[1], button=button)
    time.sleep(0.05)
    return True


def _center_on_base(hwnd: int):
    """Клик по миникарте у базы — камера на базу (выделение НЕ сбрасывается)."""
    bx, by = _minimap_client(_current_base_cell)
    _post_click(hwnd, bx, by)
    time.sleep(0.1)


_gold_region = None


def _read_gold(win_x: int, win_y: int):
    """Текущее золото (синхронный OCR региона из resource_regions.json). None если не прочиталось."""
    global _gold_region
    if _gold_region is None:
        from config import RESOURCE_REGIONS_FILE
        try:
            _gold_region = (json.loads(RESOURCE_REGIONS_FILE.read_text()).get("gold")
                            if RESOURCE_REGIONS_FILE.exists() else {})
        except Exception:
            _gold_region = {}
    if not _gold_region:
        return None
    from env.reward import _ocr_number   # тот же мульти-пороговый OCR цифр
    for _ in range(3):                   # повтор: иногда OCR с первого раза пустой
        g = _ocr_number(grab_region_bgr(win_x, win_y, _gold_region))
        if g is not None:
            return g
        time.sleep(0.05)
    return None


def _tree_mask(bgr: np.ndarray) -> np.ndarray:
    """Маска леса = ТЁМНО-зелёные насыщенные пиксели (см. TREE_GREEN_HSV_* в config)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(TREE_GREEN_HSV_LOW, np.uint8),
                            np.array(TREE_GREEN_HSV_HIGH, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


def _locate_trees(win_x: int, win_y: int) -> tuple | None:
    """
    Точка В ГУЩЕ крупнейшего «зелёного И текстурного» кластера (лес), клиентские
    координаты. None если леса в кадре нет.

    Берём НЕ центр кластера (он может попасть в прогалину → рабочий встанет, но не
    начнёт рубить), а самую «глубокую» точку (distance transform) — клик по ней
    гарантированно попадает по дереву.
    """
    bgr = grab_region_bgr(win_x, win_y, BUILDING_SEARCH_REGION)
    mask = _tree_mask(bgr)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best_i, best_area = -1, 0
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a > best_area:
            best_area, best_i = a, i
    if best_i < 0 or best_area < MIN_TREE_AREA:
        return None

    cluster = (labels == best_i).astype(np.uint8) * 255
    # Обнуляем кромку кадра: глубокая точка должна быть ВНУТРИ обзора, а не на самом
    # краю экрана (там лес обрезан границей — рабочий встаёт у края, не дойдя до дерева).
    b = 30
    cluster[:b, :] = 0
    cluster[-b:, :] = 0
    cluster[:, :b] = 0
    cluster[:, -b:] = 0
    dist = cv2.distanceTransform(cluster, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    if max_val < 4:                      # лес только у самого края кадра — пропускаем
        return None
    return (BUILDING_SEARCH_REGION["x"] + int(max_loc[0]),
            BUILDING_SEARCH_REGION["y"] + int(max_loc[1]))


def _send_workers_to(hwnd, win_x, win_y, locate, label, n_workers: int = 5):
    """
    Шлёт праздных рабочих на ресурс. `locate(win_x, win_y)` → (x,y) ресурса или None.

    Надёжность: backtick прыгает камерой на рабочего, поэтому КАЖДУЮ итерацию
    возвращаем камеру на базу И ЗАНОВО ищем ресурс. Кликаем ТОЛЬКО если ресурс
    реально в кадре — иначе пропускаем (рабочий не улетает в край карты).
    """
    _center_on_base(hwnd)
    if locate(win_x, win_y) is None:
        print(f"    [vision] {label} не найден(а) рядом с базой", flush=True)
        return

    for _ in range(n_workers):
        _key(IDLE_WORKER_KEY)        # выбрать праздного (камера прыгает к нему)
        time.sleep(0.08)
        _center_on_base(hwnd)        # вернуть камеру на базу
        time.sleep(0.22)             # дать камере устаканиться
        tgt = locate(win_x, win_y)
        if tgt is not None:          # кликаем ТОЛЬКО по подтверждённому ресурсу
            _post_click(hwnd, tgt[0], tgt[1], button="right")
        time.sleep(0.2)             # дать рабочему уйти из «праздных»


def _gather_gold(hwnd: int, win_x: int, win_y: int):
    _send_workers_to(hwnd, win_x, win_y,
                     lambda wx, wy: _locate_building(wx, wy, "goldmine",
                                                     RESOURCE_MATCH_THRESHOLD),
                     "шахта")


def _gather_wood(hwnd: int, win_x: int, win_y: int):
    _send_workers_to(hwnd, win_x, win_y, _locate_trees, "лес")


def _train(hwnd: int, win_x: int, win_y: int, building: str, key: str):
    """
    Найм/апгрейд: выделить нужное здание (по экрану) → нажать клавишу.
    Если шаблон не найден/совпадение слабое — мягкий откат: просто жмём клавишу
    (вдруг нужное здание уже выделено), поведение не хуже прежнего.
    """
    if BUILDING_CENTER_ON_BASE:
        # Центрируем камеру на базе — здания у базы попадут в кадр. Клик по
        # миникарте двигает камеру, но НЕ снимает текущее выделение.
        bx, by = _minimap_client(_current_base_cell)
        _post_click(hwnd, bx, by)
        time.sleep(0.1)

    if not _find_and_click_building(hwnd, win_x, win_y, building):
        if not _has_any_template(building):
            print(f"    [vision] нет шаблона для '{building}' — запусти: "
                  f"python calibrate.py buildings", flush=True)

    if building == "altar":
        # Герой: подтверждаем найм падением золота (как с постройкой)
        global _last_hero_hired
        g0 = _read_gold(win_x, win_y)
        _key(key)
        time.sleep(0.3)
        g1 = _read_gold(win_x, win_y)
        if g0 is not None and g1 is not None and g1 < g0 - 30:
            _last_hero_hired = True
            print(f"    ✓ герой нанят! (золото {g0}→{g1})", flush=True)
    else:
        _key(key)


# ── Названия для лога ────────────────────────────────────────────────────────

ACTION_NAMES = {
    **{i: f"атака→[{i}]" for i in range(9)},
    **{i: f"камера→[{i-9}]" for i in range(9, 18)},
    18: "выделить юнитов",
    19: "выделить рабочего",
    # Обучение юнитов (нужно выделить здание)
    20: "обучить крестьянина",      # Цитадель → P
    21: "обучить пехотинца",        # Казармы  → F
    22: "обучить стрелка",          # Казармы  → R
    # Постройки — базовые
    23: "строить ферму",            # B → F
    24: "строить казармы",          # B → B
    25: "строить лесопилку",        # B → L
    # Постройки базовые (B → key)
    26: "строить алтарь героев",    # B → A
    27: "строить сторож. башню",    # B → W
    28: "строить кузницу",          # B → S
    # Постройки продвинутые (B → V → key)
    29: "строить мастерскую",       # B → V → W
    30: "строить магич. лавку",     # B → V → V
    31: "строить тайн. святилище",  # B → V → R
    32: "строить башню грифонов",   # B → V → G
    # Юниты из продвинутых зданий
    33: "обучить волшебницу",       # Храм Истины → S
    34: "обучить ведьмака",         # Храм Истины → B
    35: "обучить рыцаря",           # Казармы → K (нужен Замок+Лесопилка+Кузница)
    # Герои из Алтаря Королей
    36: "призвать Паладина",        # Алтарь → P
    37: "призвать Горн. Короля",    # Алтарь → M
    38: "призвать Верх. мага",      # Алтарь → A
    39: "призвать Чародея Крови",   # Алтарь → B (TFT)
    # Рабочий и прочее
    40: "рабочих→золото",
    41: "улучшить→крепость",        # Цитадель → U
    42: "улучшить→замок",           # Крепость → U
    # Мастерская (Workshop)
    43: "обучить летат. машину",    # Workshop → M
    44: "обучить мортиру",          # Workshop → C
    45: "обучить осадн. машину",    # Workshop → T
    46: "ничего",
    # Ратуша
    47: "призвать ополчение",       # Town Hall → M
    # Тайное святилище (Arcane Sanctuary)
    48: "обучить жреца",            # Arcane Sanctuary → P
    # Башня грифонов (Gryphon Aviary)
    49: "обучить дракончика",       # Gryphon Aviary → D
    50: "обучить грифона",          # Gryphon Aviary → G
    # Постройка Ратуши
    51: "строить ратушу",           # B → T
    # Одиночные клавиши (контекстно-зависимые)
    52: "клавиша W",                # вод. элементаль / улучш. добычи дерева
    53: "клавиша V",                # Аватар (Горн. Король)
    54: "клавиша E",                # Изгнание (Кров. маг) / Осветит. снаряд
    55: "клавиша N",                # Призыв Феникса (Кров. маг)
    56: "клавиша L",                # Улучш. добыча дерева (Лесопилка)
    57: "рабочих→дерево",           # праздных рабочих в лес
}


# ── Главная функция ──────────────────────────────────────────────────────────

def execute_action(action_idx: int, win_x: int, win_y: int):
    """
    Выполняет действие.
    win_x, win_y — координаты окна WC3 (используются для PostMessage).
    """
    global _last_build_placed, _last_hero_hired
    _last_build_placed = False          # сбрасываем флаги на каждое действие
    _last_hero_hired = False

    hwnd = get_wc3_hwnd()
    if hwnd is None:
        return

    print(f"  {ACTION_NAMES.get(action_idx, '?')}", flush=True)

    # Найм юнитов / апгрейды: сперва ВЫДЕЛИТЬ нужное здание (по экрану), потом хоткей.
    if action_idx in ACTION_BUILD_TRAIN:
        building, key = ACTION_BUILD_TRAIN[action_idx]
        _train(hwnd, win_x, win_y, building, key)
        time.sleep(ACTION_DELAY)
        return

    if 0 <= action_idx <= 8:
        # Attack-move: нажать A, потом кликнуть на миникарте
        _key("a")
        cx, cy = _minimap_client(action_idx)
        _post_click(hwnd, cx, cy)

    elif 9 <= action_idx <= 17:
        # Камера: кликнуть на миникарте
        cx, cy = _minimap_client(action_idx - 9)
        _post_click(hwnd, cx, cy)

    elif action_idx == 18:
        _key("tab")

    elif action_idx == 19:
        _key(IDLE_WORKER_KEY)

    elif action_idx == 23:
        _build(hwnd, win_x, win_y, "f")          # ферма

    elif action_idx == 24:
        _build(hwnd, win_x, win_y, "b")          # казармы

    elif action_idx == 25:
        _build(hwnd, win_x, win_y, "l")          # лесопилка

    elif action_idx == 26:
        _build(hwnd, win_x, win_y, "a")          # алтарь героев

    elif action_idx == 27:
        _build(hwnd, win_x, win_y, "w")          # сторожевая башня

    elif action_idx == 28:
        _build(hwnd, win_x, win_y, "s")          # кузница

    elif action_idx == 29:
        _build(hwnd, win_x, win_y, "v", "w")     # мастерская (продвинутая)

    elif action_idx == 30:
        _build(hwnd, win_x, win_y, "v", "v")     # магич. лавка (продвинутая)

    elif action_idx == 31:
        _build(hwnd, win_x, win_y, "v", "r")     # тайн. святилище (продвинутая)

    elif action_idx == 32:
        _build(hwnd, win_x, win_y, "v", "g")     # башня грифонов (продвинутая)

    elif action_idx == 40:
        _gather_gold(hwnd, win_x, win_y)            # праздных рабочих в шахту

    # action_idx == 46: ничего не делать

    elif action_idx == 51:
        _build(hwnd, win_x, win_y, "t")          # восстановить ратушу

    elif action_idx == 52:
        _key("w")   # вод. элементаль (Верх. маг) / ускор. добыча дерева

    elif action_idx == 53:
        _key("v")   # Аватар (Горн. Король)

    elif action_idx == 54:
        _key("e")   # Изгнание (Кров. маг) / осветит. снаряд (Мастерская)

    elif action_idx == 55:
        _key("n")   # Призыв Феникса (Кров. маг)

    elif action_idx == 56:
        _key("l")   # Улучш. добыча дерева (Лесопилка)

    elif action_idx == 57:
        _gather_wood(hwnd, win_x, win_y)            # праздных рабочих в лес

    time.sleep(ACTION_DELAY)
