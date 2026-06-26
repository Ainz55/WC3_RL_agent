import argparse
import time
import keyboard
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from env.wc3_env import WC3Env
from config import (
    TOTAL_TIMESTEPS, N_STEPS, BATCH_SIZE, N_EPOCHS,
    LEARNING_RATE, GAMMA, ENT_COEF, CLIP_RANGE,
    MODELS_DIR, LOGS_DIR, PROJECT_DIR,
)

STOP_FILE = PROJECT_DIR / "stop.txt"
WINDOW_POS_FILE = PROJECT_DIR / "window_pos.json"
# Статистика нормализации награды (VecNormalize) — нужна для дообучения --resume
VECNORM_FILE = MODELS_DIR / "wc3_vecnormalize.pkl"


def _restore_window():
    """Восстанавливает позицию/размер окна WC3 из window_pos.json (если файл есть)."""
    import json
    import win32gui
    from config import WINDOW_TITLE

    if not WINDOW_POS_FILE.exists():
        return

    with open(WINDOW_POS_FILE) as f:
        d = json.load(f)

    found = []
    def _cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and WINDOW_TITLE in win32gui.GetWindowText(hwnd):
            found.append(hwnd)
    win32gui.EnumWindows(_cb, None)

    if not found:
        return

    win32gui.MoveWindow(found[0], d["x"], d["y"], d["w"], d["h"], True)
    print(f"[WC3] Окно: позиция ({d['x']}, {d['y']}), размер {d['w']}×{d['h']})")


class StopFileCallback(BaseCallback):
    """Останавливает обучение если появился файл stop.txt в папке проекта."""

    def __init__(self, stop_file: Path, model_dir: Path):
        super().__init__()
        self.stop_file = stop_file
        self.model_dir = model_dir

    def _on_step(self) -> bool:
        if self.stop_file.exists():
            print(f"\n[StopFile] Найден {self.stop_file.name} — останавливаю...")
            self.model.save(str(self.model_dir / "wc3_ppo_interrupted"))
            vecnorm = self.model.get_vec_normalize_env()
            if vecnorm is not None:
                vecnorm.save(str(VECNORM_FILE))
            print("Модель сохранена: wc3_ppo_interrupted.zip")
            print("Продолжить: python train.py --resume")
            self.stop_file.unlink()   # удаляем stop.txt
            return False              # False = стоп
        return True


def make_env():
    env = WC3Env()
    env = Monitor(env, str(LOGS_DIR / "monitor"))
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    # Удаляем старый stop.txt если остался с прошлого раза
    if STOP_FILE.exists():
        STOP_FILE.unlink()

    # Восстанавливаем позицию и размер окна WC3 если сохранены
    _restore_window()

    print("Запуск через 10 секунд — переключись на WC3...")
    for i in range(10, 0, -1):
        print(f"  {i}...", end="\r", flush=True)
        time.sleep(1)
    print("Поехали!                ")

    env = DummyVecEnv([make_env])

    # Нормализуем НАГРАДУ (не наблюдение — там картинки). Стабилизирует PPO при
    # разном масштабе наград (золото даёт сотые доли, победа — сотню).
    if args.resume and VECNORM_FILE.exists():
        env = VecNormalize.load(str(VECNORM_FILE), env)
        env.training = True
        env.norm_reward = True
        print(f"Загружена статистика нормализации: {VECNORM_FILE.name}")
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True,
                           clip_reward=10.0, gamma=GAMMA)

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=str(MODELS_DIR),
        name_prefix="wc3_ppo",
        save_vecnormalize=True,
    )
    stop_cb = StopFileCallback(STOP_FILE, MODELS_DIR)

    latest_zip = sorted(MODELS_DIR.glob("wc3_ppo_*.zip"))
    model = None
    if args.resume and latest_zip:
        model_path = str(latest_zip[-1])
        print(f"Продолжаю с: {model_path}")
        try:
            model = PPO.load(model_path, env=env)
        except (ValueError, KeyError, AssertionError) as e:
            print("[!] Чекпоинт не загрузился — скорее всего он старого формата "
                  "(наблюдение теперь Dict).")
            print(f"    Ошибка: {e}")
            print("    Старые модели несовместимы. Запусти БЕЗ --resume (обучение с нуля)")
            print(f"    или убери прежние .zip из {MODELS_DIR}.")
            return

    if model is None:
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            ent_coef=ENT_COEF,
            clip_range=CLIP_RANGE,
            tensorboard_log=str(LOGS_DIR),
            verbose=1,
        )

    # Глобальный хоткей F8 — работает даже когда WC3 в фокусе
    def _stop():
        if not STOP_FILE.exists():
            STOP_FILE.touch()
            print("\n[F8] Остановка — сохраняю модель...")

    keyboard.add_hotkey("F8", _stop)
    print("Для остановки: нажми F8 (работает из любого окна)")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_cb, stop_cb],
            reset_num_timesteps=not args.resume,
        )
        model.save(str(MODELS_DIR / "wc3_ppo_final"))
        env.save(str(VECNORM_FILE))
        print("Обучение завершено!")

    except KeyboardInterrupt:
        print("\nCtrl+C — сохраняю модель...")
        model.save(str(MODELS_DIR / "wc3_ppo_interrupted"))
        env.save(str(VECNORM_FILE))
        print("Модель сохранена. Продолжить: python train.py --resume")


if __name__ == "__main__":
    main()
