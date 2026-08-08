import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import ppo_cfg, device


class PPONetwork(nn.Module):
    def __init__(self, state_dim=ppo_cfg.STATE_DIM, action_dim=ppo_cfg.ACTION_DIM, hidden_dim=ppo_cfg.HIDDEN_DIM):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        feat = self.shared(x)
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value


class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_dim, device):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.device = device
        self.step_idx = 0

    def add(self, obs, actions, log_probs, rewards, dones, values):
        self.obs[self.step_idx] = obs
        self.actions[self.step_idx] = actions
        self.log_probs[self.step_idx] = log_probs
        self.rewards[self.step_idx] = rewards
        self.dones[self.step_idx] = dones.astype(np.float32)
        self.values[self.step_idx] = values
        self.step_idx += 1

    def compute_gae(self, last_values, gamma, lam):
        T, N = self.rewards.shape
        adv = np.zeros((T, N), dtype=np.float32)
        lastgaelam = np.zeros(N, dtype=np.float32)

        for t in reversed(range(T)):
            nextnonterminal = 1.0 - self.dones[t]
            nextvalues = last_values if t == T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
            adv[t] = lastgaelam

        returns = adv + self.values

        # flatten
        obs = self.obs.reshape(T * N, -1)
        actions = self.actions.reshape(T * N)
        log_probs = self.log_probs.reshape(T * N)
        adv = adv.reshape(T * N)
        returns = returns.reshape(T * N)
        values = self.values.reshape(T * N)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        log_probs_t = torch.as_tensor(log_probs, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        values_t = torch.as_tensor(values, dtype=torch.float32, device=self.device)

        # нормализация
        if adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        else:
            adv_t = adv_t - adv_t.mean()

        return obs_t, actions_t, log_probs_t, adv_t, returns_t, values_t


class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        x = x.astype(np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * (self.count * batch_count / tot_count)
        new_var = M2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class PPOAgent:
    def __init__(self, state_dim=ppo_cfg.STATE_DIM, action_dim=ppo_cfg.ACTION_DIM, lr=ppo_cfg.LEARNING_RATE):
        self.network = PPONetwork(state_dim, action_dim, ppo_cfg.HIDDEN_DIM).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), weight_decay=0, lr=lr, eps=1e-5)
        self.obs_rms = RunningMeanStd(state_dim)
        self.entropy_coef = ppo_cfg.ENTROPY_COEF

        self.training_stats = {
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "total_losses": [],
            "learning_rates": [],
            "ratio_mean": [],
            "ratio_std": [],
            "ratio_frac_out": [],
        }

    @torch.no_grad()
    def act(self, obs_batch: np.ndarray):
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
        logits, values = self.network(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def update(self, buffer: RolloutBuffer, last_values: np.ndarray):
        obs_t, actions_t, old_log_probs_t, adv_t, returns_t, _ = buffer.compute_gae(
            last_values=last_values, gamma=ppo_cfg.GAMMA, lam=ppo_cfg.GAE_LAMBDA
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n = obs_t.size(0)
        batch_size = ppo_cfg.BATCH_SIZE

        ratio_means = []
        ratio_stds = []
        ratio_fracs = []

        for epoch in range(ppo_cfg.PPO_EPOCHS):
            idx = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                bi = idx[start:end]

                logits, values = self.network(obs_t[bi])
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_t[bi])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[bi])
                surr1 = ratio * adv_t[bi]
                surr2 = torch.clamp(ratio, 1 - ppo_cfg.CLIP_EPSILON, 1 + ppo_cfg.CLIP_EPSILON) * adv_t[bi]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values, returns_t[bi])
                loss = policy_loss + ppo_cfg.VALUE_LOSS_COEF * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), ppo_cfg.MAX_GRAD_NORM)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.detach().cpu())
                total_value_loss += float(value_loss.detach().cpu())
                total_entropy += float(entropy.detach().cpu())

                with torch.no_grad():
                    r = ratio.detach()
                    ratio_means.append(float(r.mean().cpu()))
                    ratio_stds.append(float(r.std().cpu()))
                    frac_out = ((r < (1 - ppo_cfg.CLIP_EPSILON)) | (r > (1 + ppo_cfg.CLIP_EPSILON))).float().mean()
                    ratio_fracs.append(float(frac_out.cpu()))

        self.training_stats["policy_losses"].append(total_policy_loss)
        self.training_stats["value_losses"].append(total_value_loss)
        self.training_stats["entropies"].append(total_entropy)
        self.training_stats["learning_rates"].append(self.optimizer.param_groups[0]["lr"])
        if ratio_means:
            self.training_stats["ratio_mean"].append(float(np.mean(ratio_means)))
            self.training_stats["ratio_std"].append(float(np.mean(ratio_stds)))
            self.training_stats["ratio_frac_out"].append(float(np.mean(ratio_fracs)))

    def save_model(self, path):
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.training_stats,
            "obs_rms": {
                "mean": self.obs_rms.mean,
                "var": self.obs_rms.var,
                "count": self.obs_rms.count,
            },
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_stats = checkpoint.get("training_stats", self.training_stats)

        if "obs_rms" in checkpoint:
            self.obs_rms.mean = checkpoint["obs_rms"]["mean"]
            self.obs_rms.var = checkpoint["obs_rms"]["var"]
            self.obs_rms.count = checkpoint["obs_rms"]["count"]

        self.network.eval()
