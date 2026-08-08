import csv
import json
import os
import time
from datetime import datetime
from typing import List

import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv

from .config import ppo_cfg, device, game_cfg, debug_cfg, MODELS_ROOT
from .gym_env import make_env_seed_offset
from .ppo_agent import PPOAgent, RolloutBuffer
from .utils import create_next_model_run_dir


def train_agent_vectorized(
        n_envs=12,
        total_timesteps=25_000_000,
        save_interval=5_000_000,
        early_stopping_patience=5000,
        lr=ppo_cfg.LEARNING_RATE,
        lr_decay=ppo_cfg.LEARNING_RATE_DECAY
):
    run_id, run_dir = create_next_model_run_dir(MODELS_ROOT)

    print("Запуск векторизованного PPO обучения")
    print(f"Устройство: {device}")
    print(f"Количество окружений: {n_envs}")
    print(f"Base LR: {lr}")
    print(f"Всего шагов (суммарно по всем env): {total_timesteps}")

    vec_env = AsyncVectorEnv([make_env_seed_offset(i) for i in range(n_envs)])
    agent = PPOAgent(lr=lr)

    obs, info = vec_env.reset(seed=game_cfg.DEFAULT_SEED)
    training_start_time = time.time()

    csv_path = run_dir / "train_progress.csv"
    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow([
            "global_step",
            "episode_idx",
            "env_idx",
            "game_score",
            "length",
            "best_score",
            "episode_reward",
            "lr",
            "policy_loss",
            "value_loss",
            "entropy",
        ])

    episode_scores: List[float] = []
    episode_rewards: List[float] = []
    best_score = 0.0
    episode_count = 0
    episodes_without_improvement = 0
    global_step = 0
    start_time = time.time()

    best_model_path = ""
    early_stopped = False

    prev_saved_best_score = 0.0
    prev_lr = lr
    printed_score_debug = False

    while global_step < total_timesteps:
        remaining = total_timesteps - global_step
        steps_this_rollout = min(ppo_cfg.N_STEPS, (remaining + n_envs - 1) // n_envs)
        buffer = RolloutBuffer(steps_this_rollout, n_envs, ppo_cfg.STATE_DIM, device)

        for _ in range(steps_this_rollout):
            agent.obs_rms.update(obs)
            norm_obs = agent.obs_rms.normalize(obs)

            actions, log_probs, values = agent.act(norm_obs)
            next_obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            buffer.add(
                obs=norm_obs,
                actions=actions,
                log_probs=log_probs,
                rewards=rewards.astype(np.float32),
                dones=dones.astype(np.float32),
                values=values.astype(np.float32)
            )

            if isinstance(infos, dict) and "episode" in infos and "_episode" in infos:
                ep_mask = infos["_episode"]
                ep_r = infos["episode"]["r"]
                ep_l = infos["episode"]["l"]

                final_infos = infos.get("final_info", None)
                final_mask = infos.get("_final_info", None)

                done_indices = np.where(ep_mask)[0]
                for env_idx in done_indices:
                    episode_count += 1

                    episode_reward = float(ep_r[env_idx])
                    length = int(ep_l[env_idx])

                    game_score = None

                    # 1) Предпочитаем финальный info до autoreset
                    if final_infos is not None and final_mask is not None and final_mask[env_idx]:
                        fi = final_infos[env_idx]
                        if fi is not None and isinstance(fi, dict):
                            if "score" in fi:
                                game_score = float(fi["score"])
                            elif not printed_score_debug:
                                print(f"[debug] final_info keys for env {env_idx}: {list(fi.keys())}")
                                printed_score_debug = True

                    # 2) Фоллбек: если batched infos содержит score, берём оттуда
                    if game_score is None and isinstance(infos, dict) and "score" in infos:
                        try:
                            raw_score = infos["score"][env_idx]
                            if raw_score is not None:
                                game_score = float(raw_score)
                        except Exception:
                            pass

                    logged_score = game_score if game_score is not None else -1.0

                    episode_rewards.append(episode_reward)
                    if game_score is not None:
                        episode_scores.append(game_score)

                    last_lr = agent.optimizer.param_groups[0]["lr"]
                    last_policy_loss = (
                        agent.training_stats["policy_losses"][-1]
                        if agent.training_stats["policy_losses"] else 0.0
                    )
                    last_value_loss = (
                        agent.training_stats["value_losses"][-1]
                        if agent.training_stats["value_losses"] else 0.0
                    )
                    last_entropy = (
                        agent.training_stats["entropies"][-1]
                        if agent.training_stats["entropies"] else 0.0
                    )

                    csv_writer.writerow([
                        global_step,
                        episode_count,
                        env_idx,
                        logged_score,
                        length,
                        best_score,
                        episode_reward,
                        last_lr,
                        last_policy_loss,
                        last_value_loss,
                        last_entropy
                    ])
                    csv_file.flush()

                    if debug_cfg.LOG_LR and prev_lr != last_lr:
                        print(f"Изменение lr: старый lr: {prev_lr}, новый lr: {last_lr}")
                        prev_lr = last_lr

                    if game_score is not None and game_score > best_score:
                        best_score = float(game_score)
                        best_model_path = str(run_dir / f"vec_best_score_{int(best_score)}.pth")
                        episodes_without_improvement = 0

                        if game_score > 3000 and (game_score - prev_saved_best_score) > 100:
                            prev_saved_best_score = game_score
                            agent.save_model(best_model_path)
                            print(f"Новый рекорд! Счет: {best_score:.0f}, модель: {best_model_path}")
                        else:
                            print(f"Новый рекорд! Счет: {best_score:.0f}")
                    else:
                        episodes_without_improvement += 1

            obs = next_obs

        global_step += n_envs * steps_this_rollout
        current_step = n_envs * steps_this_rollout

        norm_last_obs = agent.obs_rms.normalize(obs)
        with torch.no_grad():
            obs_t = torch.as_tensor(norm_last_obs, dtype=torch.float32, device=device)
            _, next_values_t = agent.network(obs_t)
        next_values = next_values_t.detach().cpu().numpy().astype(np.float32)

        progress = min(1.0, float(global_step) / float(total_timesteps))

        base_lr = lr
        current_lr = base_lr * (1.0 - lr_decay * progress)
        for pg in agent.optimizer.param_groups:
            pg["lr"] = current_lr

        agent.entropy_coef = float(
            np.interp(progress, [0.0, 1.0], [ppo_cfg.ENTROPY_COEF, 0.005])
        )

        agent.update(buffer, last_values=next_values)

        if episode_count > 0 and (episode_count % 10 == 0 or (time.time() - start_time) > 10.0):
            avg_score = (
                float(np.mean(episode_scores[-10:])) if len(episode_scores) >= 10
                else (float(np.mean(episode_scores)) if episode_scores else -1.0)
            )
            avg_epr = (
                float(np.mean(episode_rewards[-10:])) if len(episode_rewards) >= 10
                else (float(np.mean(episode_rewards)) if episode_rewards else 0.0)
            )
            elapsed = time.time() - start_time
            sps = current_step / elapsed if elapsed > 0 else 0.0

            rm = agent.training_stats["ratio_mean"][-1] if agent.training_stats["ratio_mean"] else 1.0
            rf = agent.training_stats["ratio_frac_out"][-1] if agent.training_stats["ratio_frac_out"] else 0.0

            print(
                f"Шаг {global_step:8d} | Эпизодов: {episode_count:5d} | "
                f"Avg Score: {avg_score:7.2f} | Best: {int(best_score):4d} | Avg EPR: {avg_epr:7.2f}"
                f" | SPS: {sps:.0f} | r_mean: {rm:.3f} | frac_out: {rf:.2f}"
            )

            start_time = time.time()

        if (global_step // save_interval) != ((global_step - n_envs * steps_this_rollout) // save_interval):
            ckpt_path = str(run_dir / f"vec_checkpoint_step_{global_step}.pth")
            agent.save_model(ckpt_path)
            print(f"Checkpoint сохранен: {ckpt_path}")

        if episodes_without_improvement >= early_stopping_patience:
            print(f"\n Early stopping: {episodes_without_improvement} эпизодов без улучшения")
            early_stopped = True
            break

    final_path = str(run_dir / "vec_final_model.pth")
    agent.save_model(final_path)

    print("=" * 60)
    print("Обучение завершено!")
    print("=" * 60)
    print(f" Лучший счет: {best_score}")
    print(f" Всего эпизодов: {len(episode_rewards)}")
    print(f" Всего шагов (факт): {global_step}")
    print(f" Модель сохранена: {final_path}")
    print("=" * 60)

    training_data = {
        "best_score": best_score,
        "total_episodes": len(episode_rewards),
        "total_steps": global_step,
        "n_envs": n_envs,
        "config": {
            "gamma": ppo_cfg.GAMMA,
            "gae_lambda": ppo_cfg.GAE_LAMBDA,
            "clip_eps": ppo_cfg.CLIP_EPSILON,
            "epochs": ppo_cfg.PPO_EPOCHS,
            "batch": ppo_cfg.BATCH_SIZE,
            "n_steps": ppo_cfg.N_STEPS,
        }
    }

    with open(run_dir / "vec_training_log.json", "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2)
    print(f"Статистика сохранена: {run_dir}/vec_training_log.json")

    registry_path = os.path.join("logs", "trained_models.csv")
    registry_exists = os.path.exists(registry_path)

    param_count = int(sum(p.numel() for p in agent.network.parameters()))
    duration_sec = float(time.time() - training_start_time)
    lr_final = float(agent.optimizer.param_groups[0].get("lr", lr))

    registry_fields = [
        "finished_at",
        "best_score",
        "device",
        "seed",
        "n_envs",
        "total_timesteps_planned",
        "total_steps_done",
        "total_episodes",
        "lr_initial",
        "lr_final",
        "save_interval",
        "early_stopping_patience",
        "early_stopped",
        "final_model_path",
        "best_model_path",
        "state_dim",
        "action_dim",
        "hidden_dim",
        "param_count",
        "ppo_gamma",
        "ppo_gae_lambda",
        "ppo_clip_epsilon",
        "ppo_epochs",
        "ppo_batch_size",
        "ppo_n_steps",
        "entropy_coef_start",
        "entropy_coef_end",
        "duration_sec",
        "lr_decay",
    ]

    registry_row = {
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_score": float(best_score),
        "device": str(device),
        "seed": int(game_cfg.DEFAULT_SEED),
        "n_envs": int(n_envs),
        "total_timesteps_planned": int(total_timesteps),
        "total_steps_done": int(global_step),
        "total_episodes": int(len(episode_rewards)),
        "lr_initial": float(lr),
        "lr_final": lr_final,
        "save_interval": int(save_interval),
        "early_stopping_patience": int(early_stopping_patience),
        "early_stopped": bool(early_stopped),
        "final_model_path": final_path,
        "best_model_path": best_model_path,
        "state_dim": int(ppo_cfg.STATE_DIM),
        "action_dim": int(ppo_cfg.ACTION_DIM),
        "hidden_dim": int(ppo_cfg.HIDDEN_DIM),
        "param_count": param_count,
        "ppo_gamma": float(ppo_cfg.GAMMA),
        "ppo_gae_lambda": float(ppo_cfg.GAE_LAMBDA),
        "ppo_clip_epsilon": float(ppo_cfg.CLIP_EPSILON),
        "ppo_epochs": int(ppo_cfg.PPO_EPOCHS),
        "ppo_batch_size": int(ppo_cfg.BATCH_SIZE),
        "ppo_n_steps": int(ppo_cfg.N_STEPS),
        "entropy_coef_start": float(ppo_cfg.ENTROPY_COEF),
        "entropy_coef_end": float(getattr(agent, "entropy_coef", ppo_cfg.ENTROPY_COEF)),
        "duration_sec": duration_sec,
        "lr_decay": lr_decay,
    }

    with open(registry_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=registry_fields)
        if not registry_exists:
            w.writeheader()
        w.writerow(registry_row)
    print(f"Реестр моделей обновлен: {registry_path}")

    vec_env.close()
    csv_file.close()
    return agent

