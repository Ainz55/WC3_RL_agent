import sys
import time
import cv2
import numpy as np
import pydirectinput
import win32api
import win32gui
from pathlib import Path

from env.screen import ScreenCapture, find_game_window
from config import TEMPLATES_DIR, PROJECT_DIR

pydirectinput.PAUSE = 0.0


def cmd_window():
    import win32gui
    from config import WINDOW_TITLE
    try:
        x, y = find_game_window()
        print(f"Game window found at screen position ({x}, {y})")

        # Также выводим точный размер клиентской области
        def _cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd) and WINDOW_TITLE in win32gui.GetWindowText(hwnd):
                rect = win32gui.GetClientRect(hwnd)
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                print(f"Client area size: {w} x {h}  → установи GAME_W={w}, GAME_H={h} в config.py")
        win32gui.EnumWindows(_cb, None)
    except RuntimeError as e:
        print(f"ERROR: {e}")


def cmd_regions():
    sc = ScreenCapture()
    sc.update_window_pos()
    mm = sc.grab_minimap()
    res = sc.grab_resource_bar()
    full = sc.grab_full()

    cv2.imshow("Minimap", cv2.resize(mm, (300, 300)))
    cv2.imshow("Resource bar", cv2.resize(res, (600, 60)))
    cv2.imshow("Full window (press any key)", cv2.resize(full, (800, 600)))
    print("Press any key in an OpenCV window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _save_template(name: str):
    sc = ScreenCapture()
    sc.update_window_pos()
    center = sc.grab_result_region()
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    path = TEMPLATES_DIR / f"{name}.png"
    cv2.imwrite(str(path), gray)
    print(f"Template saved: {path}")
    cv2.imshow(f"Saved: {name}", cv2.resize(center, (400, 300)))
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def cmd_test():
    from env.wc3_env import WC3Env
    print("Initialising environment...")
    env = WC3Env()
    obs, _ = env.reset()
    print("Observation (Dict):")
    for k, v in obs.items():
        print(f"  {k:8s}: shape={v.shape}  dtype={v.dtype}")
    print(f"Action space: {env.action_space}")
    action = env.action_space.sample()
    print(f"Stepping with random action {action}...")
    obs, reward, term, trunc, info = env.step(action)
    print(f"  reward={reward:.4f}  terminated={term}  truncated={trunc}")
    env.close()
    print("All good!")


def cmd_menu():
    """
    Записывает позиции кнопок для автоперезапуска.

    Авторестарт (победа):  exit_victory → ok_results → раса → сложность → старт
    Авторестарт (поражение): exit_defeat → ok_results → раса → сложность → старт
    """
    import json
    positions_file = PROJECT_DIR / "menu_positions.json"
    positions = {}

    # Запоминаем позицию окна WC3 в момент калибровки
    win_x, win_y = find_game_window()

    def wait_and_record(key, label):
        print(f"  Наведи мышь на [{label}]...", end=" ", flush=True)
        for i in range(5, 0, -1):
            print(f"{i}", end=" ", flush=True)
            time.sleep(1)
        abs_x, abs_y = win32api.GetCursorPos()
        # Сохраняем как ОТНОСИТЕЛЬНЫЕ координаты от окна WC3
        positions[key] = {"rx": abs_x - win_x, "ry": abs_y - win_y}
        print(f"\n  ✓ [{label}]: относительно окна ({abs_x - win_x}, {abs_y - win_y})\n")

    def click_recorded(key):
        p = positions[key]
        # Пересчитываем в абсолютные координаты через текущую позицию окна
        cur_wx, cur_wy = find_game_window()
        ax, ay = cur_wx + p["rx"], cur_wy + p["ry"]
        pydirectinput.moveTo(ax, ay)
        time.sleep(0.15)
        pydirectinput.click(ax, ay)
        time.sleep(0.5)

    def record_dropdown(key_btn, key_opt, label_btn, label_opt):
        wait_and_record(key_btn, label_btn)
        click_recorded(key_btn)          # открываем дропдаун
        time.sleep(0.4)
        wait_and_record(key_opt, label_opt)

    print("=== Запись позиций меню ===")
    print("Принцип: наведи мышь на кнопку → жди 5 сек → программа записывает.\n")

    # ── Шаг 1: кнопка выхода при ПОРАЖЕНИИ ───────────────────────────────
    print("── Шаг 1: Экран ПОРАЖЕНИЯ ──")
    print("  Используй чит: somebodysetupusthebomb")
    input("  Когда виден экран 'Вы проиграли' — нажми Enter: ")
    wait_and_record("exit_defeat", "Выйти из игры (поражение)")

    # ── Шаг 2: кнопка OK на экране результатов ───────────────────────────
    print("── Шаг 2: Экран результатов (Обзор/Войска/OK) ──")
    print("  Нажми 'Выйти из игры' вручную → появится экран со статистикой.")
    input("  Когда виден экран со статистикой и кнопкой OK — нажми Enter: ")
    wait_and_record("ok_results", "OK (экран результатов)")

    # ── Шаг 3: лобби — раса и сложность ──────────────────────────────────
    print("── Шаг 3: Лобби ──")
    print("  Нажми OK вручную → дождись лобби.")
    input("  Когда лобби открылось — нажми Enter: ")
    record_dropdown("race_btn", "race_human",
                    "Любая раса (строка ИГРОКА)", "Человек")
    record_dropdown("difficulty_btn", "difficulty_easy",
                    "Компьютер (открыть дроп)", "Слабый")
    # Примечание: сложность WC3 запоминает сама. calibrate.py записывает позиции
    # на случай если авторестарт должен явно выставлять сложность.
    # В текущей версии wc3_env.py этот шаг пропускается.
    wait_and_record("start_game", "Начать игру")

    # ── Шаг 4: кнопка выхода при ПОБЕДЕ ──────────────────────────────────
    print("── Шаг 4: Экран ПОБЕДЫ ──")
    print("  Нажми 'Начать игру' вручную → сыграй и используй чит: allyourbasearebelongtous")
    print("  Если чит не работает — уничтожь главное здание врага вручную.")
    input("  Когда виден экран 'Победа!' с двумя кнопками — нажми Enter: ")
    wait_and_record("exit_victory", "Выйти из игры (победа, НИЖНЯЯ кнопка)")

    with open(positions_file, "w") as f:
        json.dump(positions, f, indent=2)
    print(f"Готово! Сохранено в {positions_file}")
    print("Агент будет сам запускать новые игры с расой Human и сложностью Слабый.")


def cmd_resources():
    """
    Записывает позиции цифр ресурсов (золото, дерево, еда) для OCR.
    Регион 80×22 пикселя вокруг каждого числа.
    """
    import json
    from config import RESOURCE_REGIONS_FILE

    sc = ScreenCapture()
    sc.update_window_pos()

    RW, RH = 80, 22   # размер региона вокруг числа

    def wait_and_record(label):
        print(f"  Наведи мышь на ЦИФРУ [{label}]...", end=" ", flush=True)
        for i in range(5, 0, -1):
            print(f"{i}", end=" ", flush=True)
            time.sleep(1)
        ax, ay = win32api.GetCursorPos()
        # Конвертируем в координаты относительно окна
        rx = ax - sc._win_x - RW // 2
        ry = ay - sc._win_y - RH // 2
        print(f"\n  ✓ [{label}]: окно-относительно x={rx}, y={ry}, w={RW}, h={RH}\n")
        return {"x": rx, "y": ry, "w": RW, "h": RH}

    print("=== Запись позиций цифр ресурсов ===")
    print("Зайди в игру — должны быть видны числа золота, дерева и еды.\n")

    regions = {}
    regions["gold"]   = wait_and_record("Золото (число)")
    regions["lumber"] = wait_and_record("Дерево (число)")
    regions["food"]   = wait_and_record("Еда — ТОЛЬКО текущее число (до слеша)")

    # Показываем как выглядят захваченные регионы
    print("Проверяю захват регионов...")
    for name, r in regions.items():
        raw = sc._sct.grab({
            "left": sc._win_x + r["x"],
            "top":  sc._win_y + r["y"],
            "width": r["w"], "height": r["h"],
        })
        img = np.array(raw, dtype=np.uint8)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"{name} ({r['w']}x{r['h']})", cv2.resize(img, (240, 66)))

    print("Окна с регионами открыты — проверь что цифры видны. Нажми любую клавишу.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    with open(RESOURCE_REGIONS_FILE, "w") as f:
        json.dump(regions, f, indent=2)
    print(f"Сохранено в {RESOURCE_REGIONS_FILE}")


def cmd_test_input():
    """
    Проверяет доходят ли нажатия клавиш до WC3.
    Нажимает пробел (Stop) — в WC3 это останавливает юнитов.
    Смотри на экран: если выделен юнит и он остановился — всё работает.
    """
    import win32gui
    from env.actions import focus_game_window
    import pydirectinput as pdi
    pdi.PAUSE = 0.0

    print("Тест ввода — нажимаю пробел в WC3 через 3 секунды...")
    print("Выдели любого юнита в игре прямо сейчас!")
    time.sleep(3)

    focus_game_window()
    time.sleep(0.3)

    fg = win32gui.GetWindowText(win32gui.GetForegroundWindow())
    print(f"Активное окно: '{fg}'")

    if "Warcraft" in fg:
        print("✓ WC3 в фокусе — отправляю пробел (Stop)...")
        pdi.press("space")
        time.sleep(0.5)
        print("Если юнит остановился — ввод работает!")
    else:
        print("✗ WC3 НЕ в фокусе — клавиши не дойдут до игры")
        print("  Попробуй вручную кликнуть на WC3 и сразу запусти этот тест снова")


def cmd_save_window():
    """Сохраняет текущую позицию и размер окна WC3 в window_pos.json."""
    import json
    from config import WINDOW_TITLE

    hwnd = None
    def _cb(h, _):
        nonlocal hwnd
        if win32gui.IsWindowVisible(h) and WINDOW_TITLE in win32gui.GetWindowText(h):
            hwnd = h
    win32gui.EnumWindows(_cb, None)

    if not hwnd:
        print(f"Окно WC3 не найдено (ищу '{WINDOW_TITLE}'). Запусти игру сначала.")
        return

    x, y, right, bottom = win32gui.GetWindowRect(hwnd)
    data = {"x": x, "y": y, "w": right - x, "h": bottom - y}
    path = PROJECT_DIR / "window_pos.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Сохранено: позиция ({data['x']}, {data['y']}), размер {data['w']}×{data['h']}")
    print(f"Файл: {path}")


def cmd_restore_window():
    """Восстанавливает позицию и размер окна WC3 из window_pos.json."""
    import json
    from config import WINDOW_TITLE

    path = PROJECT_DIR / "window_pos.json"
    if not path.exists():
        print("window_pos.json не найден. Сначала: python calibrate.py save_window")
        return

    with open(path) as f:
        data = json.load(f)

    hwnd = None
    def _cb(h, _):
        nonlocal hwnd
        if win32gui.IsWindowVisible(h) and WINDOW_TITLE in win32gui.GetWindowText(h):
            hwnd = h
    win32gui.EnumWindows(_cb, None)

    if not hwnd:
        print(f"Окно WC3 не найдено (ищу '{WINDOW_TITLE}'). Запусти игру сначала.")
        return

    win32gui.MoveWindow(hwnd, data["x"], data["y"], data["w"], data["h"], True)
    print(f"Восстановлено: позиция ({data['x']}, {data['y']}), размер {data['w']}×{data['h']}")


def cmd_buildings():
    """
    Захватывает шаблоны зданий для vision-выбора перед наймом юнитов.

    Использование:
        python calibrate.py buildings                    # список по умолчанию
        python calibrate.py buildings townhall barracks  # только эти

    Наведи мышь на ЦЕНТР здания на поле боя (НЕ на иконку в командной панели!)
    → жди 10 сек → шаблон сохраняется в templates/buildings/<name>.png.
    Имена должны совпадать с ACTION_BUILD_TRAIN в env/actions.py:
      townhall, barracks, altar, workshop, arcane, aviary
    """
    from env.screen import grab_region_gray
    from config import BUILDINGS_DIR, BUILDING_TEMPLATE_SIZE

    RU = {
        "townhall": "Ратуша — 1 ур. (главное здание)",
        "keep":     "Крепость — 2 ур. (после 1-го апгрейда Ратуши)",
        "castle":   "Замок — 3 ур. (после 2-го апгрейда)",
        "barracks": "Казармы",
        "altar":    "Алтарь Королей",
        "workshop": "Мастерская",
        "arcane":   "Тайное святилище (жрец/волшебница)",
        "aviary":   "Птичник грифонов",
        "goldmine": "Золотая шахта (для добычи золота)",
    }

    names = sys.argv[2:] or ["townhall", "keep", "castle", "barracks", "altar",
                             "workshop", "arcane", "aviary", "goldmine"]
    win_x, win_y = find_game_window()
    S = BUILDING_TEMPLATE_SIZE

    print("=== Захват шаблонов зданий ===")
    print("Наведи мышь на ЦЕНТР здания на поле боя (НЕ на иконку в командной панели!).")
    print("Захватывай только те здания, что у тебя ПОСТРОЕНЫ. Которых нет — пропусти")
    print("(Ctrl+C) или укажи нужные явно: python calibrate.py buildings barracks altar")
    print("Ратуша: захвати keep/castle ПОЗЖЕ, когда улучшишь — "
          "напр. python calibrate.py buildings keep\n")

    for name in names:
        ru = RU.get(name, name)
        print(f"  [{name}] = {ru}: наведи мышь (10 сек)...", end=" ", flush=True)
        for i in range(10, 0, -1):
            print(i, end=" ", flush=True)
            time.sleep(1)
        ax, ay = win32api.GetCursorPos()
        # Уводим курсор в угол окна, чтобы всплывающая подсказка (название /
        # «Золото: N» у шахты) НЕ попала в шаблон — иначе template-matching не
        # сработает (подсказки при поиске нет, да и число всё время меняется).
        pydirectinput.moveTo(win_x + 10, win_y + 10)
        time.sleep(0.3)
        region = {"x": ax - win_x - S // 2, "y": ay - win_y - S // 2, "w": S, "h": S}
        gray = grab_region_gray(win_x, win_y, region)
        path = BUILDINGS_DIR / f"{name}.png"
        cv2.imwrite(str(path), gray)
        print(f"\n  ✓ сохранено {path}")
        cv2.imshow(name, cv2.resize(gray, (S * 3, S * 3)))

    print("\nПроверь окна — на каждом шаблоне здание должно быть узнаваемо.")
    print("Нажми любую клавишу для выхода.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cmd_action():
    """
    Выполняет ОДНО действие агента вручную — проверка механики отдельно от обучения.
        python calibrate.py action 19    # выделить праздного рабочего (клавиша /)
        python calibrate.py action 23    # построить ферму
        python calibrate.py action 21    # обучить пехотинца (vision выберет казармы)
    Полный список действий — ACTION_NAMES в env/actions.py.
    """
    from env.actions import execute_action, focus_game_window, ACTION_NAMES, set_base_cell
    from env.screen import ScreenCapture, detect_base_cell
    from config import BASE_MINIMAP_CELL

    if len(sys.argv) < 3:
        print("Укажи номер действия, напр.: python calibrate.py action 23")
        return
    idx = int(sys.argv[2])

    print(f"Действие {idx} = {ACTION_NAMES.get(idx, '?')}")
    print("Переключаюсь на WC3 через 3 сек — СМОТРИ на игру...")
    time.sleep(3)
    focus_game_window()
    win_x, win_y = find_game_window()

    # Определяем ячейку базы так же, как обучение (иначе центрирование уедет не туда)
    sc = ScreenCapture()
    sc.update_window_pos()
    base = detect_base_cell(sc.grab_minimap(), fallback=BASE_MINIMAP_CELL)
    set_base_cell(base)
    print(f"База (по рамке камеры): ячейка {base}  — камера ДОЛЖНА быть на твоей базе!")

    execute_action(idx, win_x, win_y)
    print("\nГотово. Что произошло в игре?")
    print("  • выделился рабочий/здание (портрет внизу слева)?")
    print("  • появился курсор постройки / здание встало?")
    print("  • юнит встал в очередь обучения?")


def cmd_snap():
    """
    Сохраняет кадр игры в файл — чтобы подобрать точные цвета (футпринт, лес).
        python calibrate.py snap fp   → футпринт постройки (настрой ВРУЧНУЮ за отсчёт)
        python calibrate.py snap tr   → текущий вид (деревья)

    Для fp: за время отсчёта в самой игре — выбери рабочего, нажми B потом F (ферма),
    и наведи призрак ПОЛОВИНОЙ на Ратушу/шахту — чтобы в кадре был и КРАСНЫЙ (нельзя),
    и зелёный (можно) футпринт. Снимок берётся как есть (никакой автоматики).
    """
    name = sys.argv[2] if len(sys.argv) > 2 else "snap"
    secs = 8 if name == "fp" else 3
    if name == "fp":
        print("=== Снимок футпринта ===")
        print("За 8 сек СДЕЛАЙ В ИГРЕ: рабочий → B → F (ферма) → наведи призрак на")
        print("границу Ратуши/шахты, чтобы было видно КРАСНЫЙ и зелёный футпринт.")
    print(f"Снимаю через {secs} сек — переключись в WC3 и приготовь вид...")
    for i in range(secs, 0, -1):
        print(f"  {i}", end=" ", flush=True)
        time.sleep(1)
    sc = ScreenCapture()
    sc.update_window_pos()
    cv2.imwrite(f"{name}.png", sc.grab_full())
    print(f"\nСохранено {name}.png")


def cmd_find():
    """
    Отладка vision-поиска: ищет шаблон в ТЕКУЩЕМ кадре, печатает лучший балл
    совпадения и сохраняет картинки. Наведи камеру так, чтобы объект был ВИДЕН.
        python calibrate.py find goldmine
        python calibrate.py find barracks
    """
    import env.actions as A
    from config import (BUILDING_SEARCH_REGION, BUILDING_MATCH_THRESHOLD,
                        RESOURCE_MATCH_THRESHOLD)
    from env.screen import grab_region_gray

    name = sys.argv[2] if len(sys.argv) > 2 else "goldmine"
    sc = ScreenCapture()
    sc.update_window_pos()
    scene = grab_region_gray(sc._win_x, sc._win_y, BUILDING_SEARCH_REGION)
    cv2.imwrite("debug_search.png", scene)

    # Лес ищется ПО ЦВЕТУ (не по шаблону)
    if name == "trees":
        from env.screen import grab_region_bgr
        bgr = grab_region_bgr(sc._win_x, sc._win_y, BUILDING_SEARCH_REGION)
        cv2.imwrite("debug_search_color.png", bgr)
        cv2.imwrite("debug_trees_mask.png", A._tree_mask(bgr))   # зелёное И текстурное
        xy = A._locate_trees(sc._win_x, sc._win_y)
        vis = bgr.copy()
        if xy is not None:
            cv2.circle(vis, (xy[0] - BUILDING_SEARCH_REGION["x"],
                             xy[1] - BUILDING_SEARCH_REGION["y"]), 22, (0, 255, 0), 3)
            print("[trees] точка выбрана (зелёный круг) — проверь, попала ли в деревья")
        else:
            print("[trees] лес не найден")
        cv2.imwrite("debug_search_match.png", vis)
        print("Сохранено: debug_search_color.png (цвет), debug_trees_mask.png (маска),")
        print("           debug_search_match.png (круг на выбранной точке)")
        return

    tmpl = A._load_building_template(name)
    if tmpl is None:
        print(f"Нет шаблона {name}.png — сними: python calibrate.py buildings {name}")
        return

    thr = RESOURCE_MATCH_THRESHOLD if name == "goldmine" else BUILDING_MATCH_THRESHOLD
    res = cv2.matchTemplate(scene, tmpl, cv2.TM_CCOEFF_NORMED)
    _, mv, _, ml = cv2.minMaxLoc(res)
    th, tw = tmpl.shape[:2]
    vis = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    color = (0, 200, 0) if mv >= thr else (0, 0, 255)
    cv2.rectangle(vis, ml, (ml[0] + tw, ml[1] + th), color, 2)
    cv2.imwrite("debug_search_match.png", vis)

    verdict = "НАЙДЕНО" if mv >= thr else "слабо/не найдено"
    print(f"[{name}] лучший балл = {mv:.2f}  (порог {thr}) → {verdict}")
    print("Сохранено:")
    print("  debug_search.png        — что видит поиск (вся зона поля боя)")
    print("  debug_search_match.png  — рамка на лучшем совпадении (зелёная=ок, красная=слабо)")


def cmd_minimap():
    """
    Отладка определения базы: сохраняет, что захватывается как миникарта, и
    красную маску игрока — чтобы понять, почему detect_base_cell ошибается.
        python calibrate.py minimap
    """
    from env.screen import detect_base_cell, player_mask
    from config import MINIMAP, PLAYER_COLOR_HSV_LOW, PLAYER_COLOR_HSV_HIGH

    sc = ScreenCapture()
    sc.update_window_pos()
    mm = sc.grab_minimap()
    cv2.imwrite("debug_minimap.png", mm)

    mask = player_mask(mm)
    cv2.imwrite("debug_minimap_player.png", mask)

    cell = detect_base_cell(mm)
    print(f"MINIMAP регион (из config): {MINIMAP}")
    print(f"Цвет игрока HSV: {PLAYER_COLOR_HSV_LOW}..{PLAYER_COLOR_HSV_HIGH}")
    print(f"detect_base_cell -> ячейка {cell}")
    print(f"пикселей цвета игрока: {int((mask > 0).sum())}")
    print("Сохранено:")
    print("  debug_minimap.png         — что считается миникартой")
    print("  debug_minimap_player.png  — маска цвета игрока (белое = твои юниты/база)")


COMMANDS = {
    "window":          cmd_window,
    "buildings":       cmd_buildings,
    "action":          cmd_action,
    "minimap":         cmd_minimap,
    "find":            cmd_find,
    "snap":            cmd_snap,
    "regions":         cmd_regions,
    "victory":         lambda: _save_template("victory"),
    "defeat":          lambda: _save_template("defeat"),
    "menu":            cmd_menu,
    "resources":       cmd_resources,
    "test":            cmd_test,
    "test_input":      cmd_test_input,
    "save_window":     cmd_save_window,
    "restore_window":  cmd_restore_window,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "window"
    if cmd not in COMMANDS:
        print(f"Unknown command '{cmd}'. Choose from: {list(COMMANDS)}")
    else:
        COMMANDS[cmd]()
