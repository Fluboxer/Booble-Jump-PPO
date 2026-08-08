import sys
import time

import numpy as np
import psutil
import pygame
import torch

from booble_jump_ppo.config import MODELS_ROOT, game_cfg, device, ppo_cfg
from booble_jump_ppo.game_env import Game
from booble_jump_ppo.ppo_agent import PPOAgent
from booble_jump_ppo.training import train_agent_vectorized
from booble_jump_ppo.utils import get_latest_model_run_dir, resolve_model_path_in_run


def play_with_agent(model_path=None):
    if model_path is None:
        run_id, run_dir = get_latest_model_run_dir(MODELS_ROOT)
        model_path = str(resolve_model_path_in_run(run_dir))
        print(f"Берем последнего агента: model_{run_id}")

    print(f"Загружаем модель: {model_path}")
    agent = PPOAgent()
    try:
        agent.load_model(model_path)
        agent.network.eval()
        print("Модель успешно загружена!")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    seed = time.time_ns()
    game = Game(headless=False, seed=seed)

    print(f"\nИгра успешно запущена! Текущий seed: {seed}")

    print("\nУправление:")
    print(" SPACE - переключение режима (ИИ <---> Человек)")
    print(" R - рестарт игры")
    print(" ESC - выход")

    ai_mode = True
    clock = pygame.time.Clock()
    last_ai_toggle = 0.0

    while True:
        if not game.handle_events():
            break

        current_time = time.time()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and current_time - last_ai_toggle > 0.3:
            ai_mode = not ai_mode
            mode_text = "ИИ играет" if ai_mode else "Человек играет"
            print(f"! {mode_text}")
            last_ai_toggle = current_time

        if not game.game_over:
            if ai_mode:
                state = game.get_state()
                if hasattr(agent, "obs_rms"):
                    state = agent.obs_rms.normalize(state[np.newaxis, :])[0]
                with torch.no_grad():
                    obs_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    logits, _ = agent.network(obs_t)
                    probs = torch.softmax(logits, dim=-1)
                    if np.random.random() < 0.9:
                        action = torch.argmax(probs, dim=-1).item()
                    else:
                        action = torch.multinomial(probs, 1).item()
                game.step(action)
            else:
                game.step(None)
        game.draw()
        clock.tick(game_cfg.FPS)
    pygame.quit()


def parse_int_human(s: str, default: int) -> int:
    s = s.strip()
    if not s:
        return default

    s = s.replace(" ", "").replace("_", "")
    mult = 1
    if s[-1] in ("k", "K"):
        mult = 1_000
        s = s[:-1]
    elif s[-1] in ("m", "M"):
        mult = 1_000_000
        s = s[:-1]

    try:
        value = int(s) * mult
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        print(f"Некорректное значение, используется значение по умолчанию: {default}")
        return default


def check_memory_for_envs(n_envs: int, per_env_gb: float = 1.5) -> bool:
    # Защита от дурака
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    need_gb = n_envs * per_env_gb + 3.5  # 3.5 Гб под основной процесс

    print(f"\nДоступно памяти: {available_gb:.2f} ГБ")
    print(f"Требуется примерно: {need_gb:.2f} ГБ под окружения")

    if need_gb > available_gb:
        print("ВНИМАНИЕ: Памяти может не хватить")
        answer = input("Продолжить несмотря на предупреждение? [y/N]: ").strip().lower()
        return answer == "y"
    return True


def ask_params_and_train():
    default_envs = 12
    default_steps = 25_000_000
    default_lr = ppo_cfg.LEARNING_RATE
    default_lr_decay = ppo_cfg.LEARNING_RATE_DECAY

    print(f"Выберете количество окружений. Больше - лучше, default = {default_envs}")
    print("Каждое окружение потребляет по 1.5 ГБ ВЫДЕЛЕННОЙ памяти + 3.5 Гб на основной процесс")
    print("(при запуске на картошке можно попытаться спастись файлом подкачки)")
    raw_envs = input("Введите количество окружений (Enter = по умолчанию): ")

    n_envs = parse_int_human(raw_envs, default=default_envs)

    # Проверка доступной памяти
    if not check_memory_for_envs(n_envs, per_env_gb=1.5):
        print("Обучение отменено пользователем из-за недостатка памяти")
        return

    print(f"\nВыберете количество шагов. Default = {default_steps:,}".replace(",", " "))
    raw_steps = input("Введите количество шагов (можно 20m, 1_000_000 и т.п., Enter = по умолчанию): ")

    total_timesteps = parse_int_human(raw_steps, default=default_steps)

    print(f"\nВыберете Learning Rate. Default = {default_lr}")
    raw_lr = input("Введите Learning Rate (Enter = по умолчанию): ").strip()

    if raw_lr == "":
        lr = default_lr
    else:
        try:
            lr = float(raw_lr)
        except ValueError:
            print("Некорректный Learning Rate, используем значение по умолчанию")
            lr = default_lr

    print(f"\nВыберете Learning Rate Decay. Default = {default_lr_decay}")
    print(f"(этот параметр определяет, насколько упадёт LR к концу обучения, при 0.9 под конец останется 10% от "
          f"изначального LR)")
    raw_lr_decay = input("Введите Learning Rate Decay (Enter = по умолчанию): ").strip()

    if raw_lr_decay == "":
        lr_decay = default_lr_decay
    else:
        try:
            lr_decay = float(raw_lr_decay)
        except ValueError:
            print("Некорректный Learning Rate Decay, используем значение по умолчанию")
            lr_decay = default_lr_decay

    print(f"\nИтоговые параметры:")
    print(f"  Окружения: {n_envs}")
    print(f"  Шаги: {total_timesteps:,}".replace(",", " "))
    print(f"  Learning Rate: {lr}")
    print(f"  Learning Rate Decay: {lr_decay}")

    confirm = input("Запустить обучение с этими параметрами? [Y/n]: ").strip().lower()
    if confirm in ("", "y", "yes", "д", "да", "+"):
        train_agent_vectorized(n_envs=n_envs, total_timesteps=total_timesteps, lr=lr, lr_decay=lr_decay)
    else:
        print("Обучение отменено")

# Идея: 1 качает/запускает модель, если нет обученной модели со стороны пользователя
if __name__ == "__main__":
    print("Введите 1 для использования демонстрационной модели агента")
    print("Введите 2 для обучения агента со стандартными настройками")
    print("Введите 3 для игры с обученным агентом")
    print("Введите 9 для обучения агента с выбором настроек")
    print("Введите 0 для выхода из программы")

    choice = input("Введите выбор: ").strip()

    if not choice:
        choice = '3'
        print("\nПустой ввод - попытка запуска игры с агентом...")

    if choice == '2':
        train_agent_vectorized(n_envs=12, total_timesteps=25_000_000)
    elif choice == '3':
        play_with_agent()
    elif choice == '9':
        ask_params_and_train()
    elif choice == '0':
        print("Выход из программы.")
        sys.exit(0)
    else:
        print("Неверный выбор, попробуйте снова.")
