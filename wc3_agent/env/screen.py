"""Screen capture utilities: find game window and grab regions."""
import numpy as np
import cv2
import mss
import win32gui
import win32con
from config import (
    WINDOW_TITLE, GAME_W, GAME_H, OBS_SIZE, MINIMAP, RESOURCE_BAR,
    PLAYER_COLOR_HSV_LOW, PLAYER_COLOR_HSV_HIGH,
)


def player_mask(bgr: np.ndarray) -> np.ndarray:
    """Бинарная маска пикселей ЦВЕТА ИГРОКА (диапазон HSV из config)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv,
                       np.array(PLAYER_COLOR_HSV_LOW, dtype=np.uint8),
                       np.array(PLAYER_COLOR_HSV_HIGH, dtype=np.uint8))


_module_sct = None


def grab_region_gray(win_x: int, win_y: int, region: dict) -> np.ndarray:
    """
    Захватывает регион (координаты относительно окна WC3) → grayscale uint8.

    Общая для поиска зданий (actions.py) и захвата их шаблонов (calibrate.py),
    чтобы и шаблон, и сцена проходили ОДИН и тот же конвейер — иначе
    template-matching не совпадёт.
    """
    global _module_sct
    if _module_sct is None:
        _module_sct = mss.mss()
    raw = _module_sct.grab({
        "left":   win_x + region["x"],
        "top":    win_y + region["y"],
        "width":  region["w"],
        "height": region["h"],
    })
    img = np.array(raw, dtype=np.uint8)[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def grab_region_bgr(win_x: int, win_y: int, region: dict) -> np.ndarray:
    """Захват региона (координаты относительно окна WC3) → BGR uint8 (как grab_full)."""
    global _module_sct
    if _module_sct is None:
        _module_sct = mss.mss()
    raw = _module_sct.grab({
        "left":   win_x + region["x"],
        "top":    win_y + region["y"],
        "width":  region["w"],
        "height": region["h"],
    })
    img = np.array(raw, dtype=np.uint8)[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def find_game_window() -> tuple[int, int]:
    """Return (left, top) of the WC3 window client area, or raise RuntimeError."""
    result = []

    def _cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and WINDOW_TITLE in win32gui.GetWindowText(hwnd):
            rect = win32gui.GetClientRect(hwnd)
            pt = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
            result.append(pt)

    win32gui.EnumWindows(_cb, None)
    if not result:
        raise RuntimeError(
            f"WC3 window not found. Make sure the game is running in windowed mode "
            f"and the title contains '{WINDOW_TITLE}'."
        )
    return result[0]  # (left, top)


class ScreenCapture:
    """Wraps mss for fast per-region screen grabs."""

    def __init__(self):
        self._sct = mss.mss()
        self._win_x = 0
        self._win_y = 0

    def update_window_pos(self):
        self._win_x, self._win_y = find_game_window()

    def _region(self, region: dict) -> dict:
        return {
            "left": self._win_x + region["x"],
            "top": self._win_y + region["y"],
            "width": region["w"],
            "height": region["h"],
        }

    def grab_minimap(self) -> np.ndarray:
        """Returns minimap as (H, W, 3) BGR uint8."""
        raw = self._sct.grab(self._region(MINIMAP))
        img = np.array(raw, dtype=np.uint8)[:, :, :3]  # drop alpha
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def grab_resource_bar(self) -> np.ndarray:
        """Returns resource bar strip as (H, W, 3) BGR uint8."""
        raw = self._sct.grab(self._region(RESOURCE_BAR))
        img = np.array(raw, dtype=np.uint8)[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def grab_result_region(self) -> np.ndarray:
        """Center region used for win/lose detection."""
        from config import RESULT_REGION
        raw = self._sct.grab(self._region(RESULT_REGION))
        img = np.array(raw, dtype=np.uint8)[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def grab_full(self) -> np.ndarray:
        """Full game window screenshot."""
        region = {"left": self._win_x, "top": self._win_y, "width": GAME_W, "height": GAME_H}
        raw = self._sct.grab(region)
        img = np.array(raw, dtype=np.uint8)[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def detect_base_cell(minimap_bgr: np.ndarray, fallback: int = 4) -> int:
    """
    Определяет ячейку 3×3 миникарты, где находится база игрока.

    Метод: на миникарте есть белая рамка-индикатор камеры. В начале матча камера
    ВСЕГДА на базе игрока, поэтому вызывать сразу после старта (env.reset()) —
    тогда положение камеры = база. Цвет игрока для этого не годится (бирюза/синий
    сливаются с террейном), а белая рамка — однозначна.

    ВНИМАНИЕ: возвращает ТЕКУЩЕЕ положение камеры. Если камеру увели от базы,
    вернёт её, а не базу.

    Возвращает индекс 0–8:
      0(верх-лево) 1(верх-центр) 2(верх-право)
      3(центр-лево)   4(центр)   5(центр-право)
      6(низ-лево)  7(низ-центр)  8(низ-право)

    При неудаче возвращает fallback.
    """
    h, w = minimap_bgr.shape[:2]

    # Белая рамка камеры: все каналы яркие
    b, g, r = cv2.split(minimap_bgr)
    white = ((b > 205) & (g > 205) & (r > 205)).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(white)
    if n_labels <= 1:                       # белой рамки не видно
        return fallback

    # Самый крупный белый кластер = рамка камеры
    best_i = max(range(1, n_labels),
                 key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    cx, cy = centroids[best_i]
    col = min(int(cx) * 3 // w, 2)
    row = min(int(cy) * 3 // h, 2)
    return row * 3 + col


def preprocess_minimap(bgr: np.ndarray, size: int = OBS_SIZE) -> np.ndarray:
    """Resize minimap to (size, size) grayscale, return as float32 [0,1]."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def preprocess_resources(bgr: np.ndarray, size: int = OBS_SIZE) -> np.ndarray:
    """
    Resize resource bar to (size, size) grayscale.
    The bar is wide but thin, so we resize to a square strip.

    Не используется в наблюдении (ресурсы теперь идут числовым вектором stats),
    оставлено для отладки/совместимости.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def preprocess_screen(bgr: np.ndarray, size: int = OBS_SIZE) -> np.ndarray:
    """
    Весь экран игры → квадрат (size, size) в оттенках серого, float32 [0,1].

    Это даёт агенту обзор поля боя, командной панели (что сейчас выделено) и
    портрета выделённого юнита — контекст, без которого контекстные хоткеи
    (обучить/строить/способности) были «вслепую».
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0
