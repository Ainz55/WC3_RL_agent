"""
Gymnasium environment wrapping WC3 Melee.

Observation space: Dict (политика MultiInputPolicy)
  - "screen"  : Box(0,255,(N_FRAMES, OBS_SIZE, OBS_SIZE), uint8) — кадры всего экрана
  - "minimap" : Box(0,255,(N_FRAMES, OBS_SIZE, OBS_SIZE), uint8) — кадры миникарты
  - "stats"   : Box(0,10,(STATS_DIM,), float32) — нормированные числовые ресурсы

Action space: Discrete(N_ACTIONS)

Episode flow:
  1. User manually starts a WC3 melee game and calls env.reset().
  2. Agent plays until win/lose detected or MAX_STEPS reached.
  3. reset() автоматически перезапускает игру через меню (нужен menu_positions.json).
"""
import json
import time
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pydirectinput

from env.screen import ScreenCapture, preprocess_minimap, preprocess_screen, detect_base_cell
from env.actions import (execute_action, N_ACTIONS, set_base_cell,
                         last_build_placed, last_hero_hired)
from env.reward import RewardDetector
from config import (
    OBS_SIZE, N_FRAMES, STATS_DIM, MAX_STEPS_PER_EPISODE, PROJECT_DIR,
    REPEAT_THRESHOLD, REPEAT_PENALTY, REPEAT_MAX_PENALTY,
    BASE_MINIMAP_CELL, BUILD_COOLDOWN_STEPS, REWARD_NEW_BUILDING, REWARD_NEW_HERO,
    REWARD_ATTACK, ATTACK_MIN_FOOD, ATTACK_REWARD_COOLDOWN,
)


def _load_menu_positions():
    path = PROJECT_DIR / "menu_positions.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _abs(pos: dict, win_x: int, win_y: int) -> tuple[int, int]:
    """Переводит относительные координаты кнопки в абсолютные координаты экрана."""
    return win_x + pos["rx"], win_y + pos["ry"]


def _click_screen(x: int, y: int):
    """Кликает по абсолютным координатам экрана."""
    pydirectinput.moveTo(x, y)
    time.sleep(0.1)
    pydirectinput.click(x, y)
    time.sleep(0.1)


class WC3Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, auto_restart: bool = True):
        super().__init__()
        self.render_mode = render_mode
        self.auto_restart = auto_restart
        self._menu_pos = _load_menu_positions()

        if auto_restart and self._menu_pos is None:
            print("[WC3Env] menu_positions.json не найден — автоперезапуск выключен.")
            print("         Запусти: python calibrate.py menu")
            self.auto_restart = False

        self.action_space = spaces.Discrete(N_ACTIONS)

        # Dict-наблюдение: экран + миникарта (картинки, frame-stack) + числовые stats
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(low=0, high=255,
                                 shape=(N_FRAMES, OBS_SIZE, OBS_SIZE), dtype=np.uint8),
            "minimap": spaces.Box(low=0, high=255,
                                  shape=(N_FRAMES, OBS_SIZE, OBS_SIZE), dtype=np.uint8),
            "stats": spaces.Box(low=0.0, high=10.0,
                                shape=(STATS_DIM,), dtype=np.float32),
        })

        self._screen = ScreenCapture()
        self._reward = RewardDetector(self._screen)

        self._screen_stack = deque(maxlen=N_FRAMES)
        self._mm_stack = deque(maxlen=N_FRAMES)
        self._step_count = 0
        self._episode_count = 0

        # Счётчик повторяющихся действий
        self._last_action: int = -1
        self._repeat_count: int = 0

        # Флаг: эпизод завершился реально (win/lose) или по таймауту
        # True  = terminated → победа/поражение → авторестарт нужен
        # False = truncated  → таймаут, игра ещё идёт → авторестарт НЕ нужен
        self._last_episode_terminated: bool = False

        # Cooldown после постройки: блокирует команды юниту чтобы
        # крестьянин успел дойти до места стройки и не получил новую команду
        self._build_cooldown: int = 0

        # Индексы build-действий (23-32 + 51)
        self._build_actions: frozenset = frozenset(range(23, 33)) | {51}
        # Действия которые отменяют приказ крестьянину (атака и правый клик)
        self._cancel_build_actions: frozenset = frozenset(range(0, 9)) | {40}

        # Действия «обучить юнита» — их НЕ штрафуем за повтор: массировать армию
        # (жать «обучить пехотинца» подряд) — это правильно, а не спам.
        # Герои (36-39) и ополчение (47) сюда НЕ входят — их массировать нельзя.
        self._train_unit_actions: frozenset = frozenset(
            {20, 21, 22, 33, 34, 35, 43, 44, 45, 48, 49, 50}
        )

        # Награда за агрессию: ячейка базы и шаг последней выданной награды
        self._base_cell: int = BASE_MINIMAP_CELL
        self._last_attack_reward_step: int = -ATTACK_REWARD_COOLDOWN

    # ── Auto-restart ────────────────────────────────────────────────────────

    def _do_auto_restart(self, last_result: str | None):
        """
        Автоперезапуск:
          1. 'Выйти из игры' (разная позиция для победы и поражения)
          2. 'OK' на экране результатов
          3. Раса → Человек
          4. Сложность → Слабый
          5. 'Начать игру'
        """
        pos = self._menu_pos
        print(f"[WC3Env] Эпизод {self._episode_count} закончен ({last_result}) — перезапускаю...")

        # Получаем ТЕКУЩУЮ позицию окна WC3 (окно могло переместиться)
        from env.screen import find_game_window
        wx, wy = find_game_window()

        DELAY = 12  # секунд ожидания (дольше = надёжнее, экран успевает появиться)

        def click(key, label=""):
            x, y = _abs(pos[key], wx, wy)
            _click_screen(x, y)
            if label:
                print(f"[Меню] Нажато: {label}")

        def wait(sec, reason=""):
            if reason:
                print(f"[Меню] Жду {sec} сек ({reason})...")
            time.sleep(sec)

        # 1. Выйти из игры
        wait(DELAY, "экран победы/поражения")
        exit_key = "exit_victory" if last_result == "win" else "exit_defeat"
        click(exit_key, "Выйти из игры")

        # 2. OK на экране результатов
        wait(DELAY, "экран результатов")
        click("ok_results", "OK")

        # 3. Раса → Человек
        wait(DELAY, "лобби — раса")
        click("race_btn", "Любая раса (открыть)")
        time.sleep(0.8)
        click("race_human", "Человек")

        # 4. Начать игру (сложность WC3 запоминает сама — выставь вручную перед первой игрой)
        wait(DELAY, "лобби — старт")
        click("start_game", "Начать игру")

        # Ждём загрузки карты
        wait(10, "загрузка карты")
        print("[WC3Env] Новая игра запущена!")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _grab_obs(self) -> dict:
        screen_bgr = self._screen.grab_full()
        mm_bgr = self._screen.grab_minimap()
        screen_f = (preprocess_screen(screen_bgr) * 255).astype(np.uint8)
        mm_f = (preprocess_minimap(mm_bgr) * 255).astype(np.uint8)
        self._screen_stack.append(screen_f)
        self._mm_stack.append(mm_f)
        return self._build_obs()

    def _pad_stacks(self, frame_screen: np.ndarray, frame_mm: np.ndarray):
        """Fill stacks with the first frame (at episode start)."""
        for _ in range(N_FRAMES):
            self._screen_stack.append(frame_screen)
            self._mm_stack.append(frame_mm)

    def _get_stats(self) -> np.ndarray:
        """
        Числовой вектор состояния (нормирован примерно к [0,1]):
          0 gold/2000        2 food_cur/100     4 food_cur/food_max (заполнение лимита)
          1 lumber/2000      3 food_max/100     5 build_cooldown/BUILD_COOLDOWN_STEPS
                                                 6 step/MAX_STEPS (прогресс эпизода)
        """
        r = self._reward.get_resources()
        gold = r.get("gold") or 0
        lumber = r.get("lumber") or 0
        food_cur = r.get("food_cur") or 0
        food_max = r.get("food_max") or 0
        cd = BUILD_COOLDOWN_STEPS if BUILD_COOLDOWN_STEPS else 1
        stats = np.array([
            gold / 2000.0,
            lumber / 2000.0,
            food_cur / 100.0,
            food_max / 100.0,
            food_cur / food_max if food_max else 0.0,
            self._build_cooldown / cd,
            self._step_count / MAX_STEPS_PER_EPISODE,
        ], dtype=np.float32)
        # Держим в границах объявленного Box (золото в долгой игре может превысить потолок)
        return np.clip(stats, 0.0, 10.0)

    def _build_obs(self) -> dict:
        return {
            "screen": np.stack(list(self._screen_stack), axis=0),
            "minimap": np.stack(list(self._mm_stack), axis=0),
            "stats": self._get_stats(),
        }

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Автоперезапуск — только если игра реально закончилась (win/lose),
        # а не по таймауту (в этом случае игра ещё идёт и меню нет)
        if self.auto_restart and self._episode_count > 0:
            if self._last_episode_terminated:
                self._do_auto_restart(getattr(self, "_last_result", None))
            else:
                print("[WC3Env] Эпизод закончился по таймауту — "
                      "игра ещё идёт, перезапуск пропущен.")
                print("         Заверши игру вручную (проиграй/выиграй) "
                      "или подожди следующей попытки.")

        self._episode_count += 1

        # Locate the game window every episode (window may have moved)
        self._screen.update_window_pos()
        self._reward.reset()
        self._step_count = 0
        self._last_action = -1
        self._repeat_count = 0
        self._build_cooldown = 0
        self._last_attack_reward_step = -ATTACK_REWARD_COOLDOWN

        # Фокусируем WC3 один раз в начале эпизода
        from env.actions import focus_game_window
        focus_game_window()
        print(f"[Эпизод {self._episode_count}] Окно WC3 сфокусировано. Начинаю играть...")

        # Grab initial frames and fill stacks
        screen_bgr = self._screen.grab_full()
        mm_bgr = self._screen.grab_minimap()
        screen_f = (preprocess_screen(screen_bgr) * 255).astype(np.uint8)
        mm_f = (preprocess_minimap(mm_bgr) * 255).astype(np.uint8)
        self._pad_stacks(screen_f, mm_f)

        # Пнуть фоновый OCR, чтобы вектор stats заполнился реальными числами
        if self._reward._res_reader is not None:
            self._reward._res_reader.request()

        # Авто-определяем ячейку базы по индикатору камеры на миникарте.
        # В начале матча камера всегда смотрит на базу игрока.
        base_cell = detect_base_cell(mm_bgr, fallback=BASE_MINIMAP_CELL)
        set_base_cell(base_cell)
        self._base_cell = base_cell
        print(f"[Эпизод {self._episode_count}] База: ячейка {base_cell} "
              f"(0=верх-лево … 8=низ-право)")

        obs = self._build_obs()
        info = {"step": 0}
        return obs, info

    def step(self, action: int):
        win_x = self._screen._win_x
        win_y = self._screen._win_y

        # Если крестьянин идёт строить — не отменяем его приказ
        prev_cooldown = self._build_cooldown
        actual_action = action
        if self._build_cooldown > 0 and action in self._cancel_build_actions:
            actual_action = 46  # do nothing вместо отмены приказа
            self._build_cooldown -= 1
        elif action in self._build_actions:
            self._build_cooldown = BUILD_COOLDOWN_STEPS
        elif self._build_cooldown > 0:
            self._build_cooldown -= 1

        execute_action(int(actual_action), win_x, win_y)

        # Capture new state
        obs = self._grab_obs()

        # Проверяем победу/поражение раз в 5 шагов (чаще = не пропускаем экран
        # поражения; иначе агент долбит на экране поражения и не перезапускает катку)
        if self._step_count % 5 == 0:
            center_bgr = self._screen.grab_result_region()
            enemy_reward = self._reward.check_enemy_kills(self._screen.grab_minimap())
        else:
            center_bgr = None
            enemy_reward = 0.0

        reward, terminated, result = self._reward.compute_reward(center_bgr)

        if result is not None:
            self._last_result = result   # запоминаем для авторестарта

        # Награда за ПОСТРОЙКУ — по подтверждённой установке (золото упало в _build),
        # а не по хрупкому детектору на миникарте. Надёжно: точно знаем, что встало.
        if last_build_placed():
            reward += REWARD_NEW_BUILDING
            print(f"  ✓ здание построено! +{REWARD_NEW_BUILDING:.1f}", flush=True)

        if last_hero_hired():
            reward += REWARD_NEW_HERO
            print(f"  ✓ герой нанят! +{REWARD_NEW_HERO:.1f}", flush=True)

        reward += enemy_reward     # прокси «убийства»: красное на миникарте уменьшилось

        # ── Награда за агрессию (прокси «урона врагу») ──────────────────────
        # Атака (attack-move 0-8) на чужую территорию, КОГДА есть армия, —
        # предпосылка нанесения урона. Ограничено кулдауном (анти-фарм).
        if 0 <= action <= 8 and action != self._base_cell:
            food_cur = self._reward.get_resources().get("food_cur") or 0
            if (food_cur >= ATTACK_MIN_FOOD and
                    self._step_count - self._last_attack_reward_step >= ATTACK_REWARD_COOLDOWN):
                reward += REWARD_ATTACK
                self._last_attack_reward_step = self._step_count

        # ── Штраф за спам одного действия ───────────────────────────────────
        if action == self._last_action:
            self._repeat_count += 1
        else:
            self._repeat_count = 0
            self._last_action = action

        # Действия «обучить юнита» НЕ штрафуем — массировать армию это нормально
        if (self._repeat_count > REPEAT_THRESHOLD
                and action not in self._train_unit_actions):
            extra = self._repeat_count - REPEAT_THRESHOLD
            repeat_penalty = max(REPEAT_PENALTY * extra, REPEAT_MAX_PENALTY)
            reward += repeat_penalty

        self._step_count += 1
        truncated = self._step_count >= MAX_STEPS_PER_EPISODE

        # Запоминаем как закончился эпизод: реально (win/lose) или таймаут
        if terminated or truncated:
            self._last_episode_terminated = terminated  # True только при win/lose

        info = {
            "step": self._step_count,
            "result": result,
            "repeat_count": self._repeat_count,
            "build_cooldown": self._build_cooldown,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            import cv2
            frame = self._screen.grab_full()
            cv2.imshow("WC3 Agent View", frame)
            cv2.waitKey(1)

    def close(self):
        import cv2
        cv2.destroyAllWindows()
