import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from CoolingEnv import CoolingEnv  # ← 반드시 환경 포함한 모듈에서 import




class EpisodeRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_temp_penalties = []
        self.episode_energy_penalties = []
        self.episode_near_overheat_penalties = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            # SB3 + Gymnasium에서는 'final_observation'/'terminal_observation' 대신 누적값에 집중!
            if (
                "total_temp_penalty" in info and
                "total_energy_penalty" in info and
                "total_near_overheat_penalty" in info
            ):
                # 에피소드 끝이면 info에 누적 값이 들어있음 (누적 변수를 step info에 항상 넣어두는 게 제일 안전)
                self.episode_temp_penalties.append(info["total_temp_penalty"])
                self.episode_energy_penalties.append(info["total_energy_penalty"])
                self.episode_near_overheat_penalties.append(info["total_near_overheat_penalty"])
            # 필요하다면 episode_rewards도 따로 저장 (SB3에서는 자동으로 저장됨)
        return True

# 환경 인스턴스 (gymnasium.Env 기반)
env = CoolingEnv()

# 모델 정의 및 학습
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=10000,
    batch_size=64,
    train_freq=4,
    target_update_interval=100,
    gamma=0.99,
    exploration_fraction=0.8,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.02,
    verbose=1
    )
callback = EpisodeRewardCallback()
model.learn(total_timesteps=1000000, callback=callback)

# 결과 시각화
plt.figure(figsize=(10, 5))
plt.plot(callback.episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("Reward per Episode")
plt.grid()
plt.show()