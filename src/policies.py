"""六个 router。全部实现 interfaces.Policy：act(obs, t) -> action index。

分三类：
- 无学习：RandomPolicy, RoundRobinPolicy, GapPolicy
- 从训练轨迹拟合：GreedyPolicy（只看即时收益）, LinUCBBandit（加探索）
- 序列学习：DQNPolicy

GreedyPolicy 和 DQNPolicy 的对比就是 proposal 的核心问题：
延迟 credit assignment 是否带来收益。bandit 夹在中间作诊断。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .interfaces import N_ACTIONS, N_OBJ


# ==========================================================================
# 无学习基线
# ==========================================================================
@dataclass
class RandomPolicy:
    seed: int = 0
    name: str = "random"

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def reset(self, episode_seed: int) -> None:
        self._rng = random.Random(self.seed * 100003 + episode_seed)

    def act(self, obs, t: int) -> int:
        return self._rng.randrange(N_ACTIONS)

    def update(self, *a, **k) -> None:
        pass


@dataclass
class RoundRobinPolicy:
    name: str = "round_robin"

    def reset(self, episode_seed: int) -> None:
        pass

    def act(self, obs, t: int) -> int:
        return t % N_ACTIONS

    def update(self, *a, **k) -> None:
        pass


@dataclass
class GapPolicy:
    """选当前分子上归一化后离目标最远的那个目标对应的 expert。

    未归一化的版本会退化成"永远调同一个 expert"：各目标的分数尺度差异很大
    （JNK3 普遍接近 0，SA 普遍接近 0.9），1-score 的排序几乎不随分子变化。
    这里用训练集上各目标的标准差做归一化。
    """

    name: str = "gap"
    target: float = 1.0
    scale: np.ndarray | None = None

    def fit(self, transitions) -> "GapPolicy":
        X = np.array([tr["obs"][:N_OBJ] for tr in transitions], dtype=float)
        # 用各目标到 1.0 的平均距离做归一化：把"普遍很低"和"这个分子特别低"分开
        self.scale = np.clip((self.target - X).mean(axis=0), 1e-3, None)
        return self

    def reset(self, episode_seed: int) -> None:
        pass

    def act(self, obs, t: int) -> int:
        cur = np.asarray(obs[:N_OBJ], dtype=float)
        s = self.scale if self.scale is not None else np.ones(N_OBJ)
        return int(np.argmax((self.target - cur) / s))

    def update(self, *a, **k) -> None:
        pass


# ==========================================================================
# 从训练轨迹拟合
# ==========================================================================
@dataclass
class GreedyPolicy:
    """每个 action 一个 ridge 回归，预测即时 reward，取 argmax。无探索。

    proposal 里的 "greedy expected-immediate-gain router fitted on
    training transitions"。它和 DQN 的差别**只在于**有没有考虑未来。
    """

    name: str = "greedy"
    alpha: float = 1.0
    W: np.ndarray | None = None

    def fit(self, transitions) -> "GreedyPolicy":
        d = len(transitions[0]["obs"])
        self.W = np.zeros((N_ACTIONS, d + 1))
        for a in range(N_ACTIONS):
            rows = [tr for tr in transitions if tr["action"] == a]
            if not rows:
                continue
            X = np.array([np.append(tr["obs"], 1.0) for tr in rows])
            y = np.array([tr["reward"] for tr in rows])
            A = X.T @ X + self.alpha * np.eye(X.shape[1])
            self.W[a] = np.linalg.solve(A, X.T @ y)
        return self

    def reset(self, episode_seed: int) -> None:
        pass

    def act(self, obs, t: int) -> int:
        x = np.append(np.asarray(obs, dtype=float), 1.0)
        return int(np.argmax(self.W @ x))

    def update(self, *a, **k) -> None:
        pass


@dataclass
class LinUCBBandit:
    """上下文 bandit。训练时带探索，评估时取 argmax（纯利用）。

    它和 GreedyPolicy 用同样的特征、同样的即时 reward，
    唯一的差别是训练期间的探索。和 DQN 的差别才是"序列性"。
    """

    name: str = "bandit"
    alpha: float = 0.5           # 探索强度
    ridge: float = 1.0
    explore: bool = False        # 评估时置 False

    A: list = field(default_factory=list)
    b: list = field(default_factory=list)

    def _init(self, d: int) -> None:
        self.A = [self.ridge * np.eye(d + 1) for _ in range(N_ACTIONS)]
        self.b = [np.zeros(d + 1) for _ in range(N_ACTIONS)]

    def fit(self, transitions) -> "LinUCBBandit":
        d = len(transitions[0]["obs"])
        self._init(d)
        for tr in transitions:
            x = np.append(np.asarray(tr["obs"], dtype=float), 1.0)
            a = tr["action"]
            self.A[a] += np.outer(x, x)
            self.b[a] += tr["reward"] * x
        return self

    def reset(self, episode_seed: int) -> None:
        pass

    def act(self, obs, t: int) -> int:
        x = np.append(np.asarray(obs, dtype=float), 1.0)
        scores = []
        for a in range(N_ACTIONS):
            Ainv = np.linalg.inv(self.A[a])
            theta = Ainv @ self.b[a]
            mu = float(theta @ x)
            if self.explore:
                mu += self.alpha * float(np.sqrt(x @ Ainv @ x))
            scores.append(mu)
        return int(np.argmax(scores))

    def update(self, obs, action, reward, *a, **k) -> None:
        x = np.append(np.asarray(obs, dtype=float), 1.0)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x


# ==========================================================================
# DQN
# ==========================================================================
class DQNPolicy:
    """小 MLP + 经验回放 + target 网络。epsilon 在训练期线性衰减。

    TA 提到的稀疏 reward / 探索问题，主要靠两点缓解：
    1. epsilon 衰减跑满训练的 60%，而不是前 10% 就降到底
    2. reward 做了尺度放大（HV 增益量级只有 1e-3，不放大 MLP 学不动）
    """

    name = "dqn"

    def __init__(self, obs_dim: int, hidden: int = 128, lr: float = 1e-3,
                 gamma: float = 0.95, reward_scale: float = 2000.0, seed: int = 0):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.q = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.epsilon = 0.0
        self._rng = random.Random(seed)

    def reset(self, episode_seed: int) -> None:
        pass

    def act(self, obs, t: int) -> int:
        import torch
        if self.epsilon > 0 and self._rng.random() < self.epsilon:
            return self._rng.randrange(N_ACTIONS)
        with torch.no_grad():
            x = torch.tensor(np.asarray(obs, dtype=np.float32))
            return int(self.q(x).argmax().item())

    def update(self, *a, **k) -> None:
        pass

    # ---- 训练 ----
    def train_on(self, buffer, n_updates: int = 2000, batch: int = 64,
                 target_every: int = 100) -> list[float]:
        import torch
        losses = []
        obs = torch.tensor(np.array([t["obs"] for t in buffer], dtype=np.float32))
        act = torch.tensor(np.array([t["action"] for t in buffer]), dtype=torch.long)
        rew = torch.tensor(
            np.array([t["reward"] for t in buffer], dtype=np.float32)
        ) * self.reward_scale
        nxt = torch.tensor(np.array([t["next_obs"] for t in buffer], dtype=np.float32))
        done = torch.tensor(np.array([t["done"] for t in buffer], dtype=np.float32))

        n = len(buffer)
        for i in range(n_updates):
            idx = torch.randint(0, n, (min(batch, n),))
            q = self.q(obs[idx]).gather(1, act[idx].unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = self.target(nxt[idx]).max(dim=1).values
                target = rew[idx] + self.gamma * (1 - done[idx]) * q_next
            loss = torch.nn.functional.smooth_l1_loss(q, target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(float(loss.item()))
            if (i + 1) % target_every == 0:
                self.target.load_state_dict(self.q.state_dict())
        return losses


def make_baselines(seed: int = 0) -> dict:
    """两个完全无学习的基线。gap 需要先 fit 归一化尺度。"""
    return {
        "random": RandomPolicy(seed=seed),
        "round_robin": RoundRobinPolicy(),
    }