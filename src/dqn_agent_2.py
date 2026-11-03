import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from CoolingEnv import CoolingEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

# 만약 CoolingEnv가 현재 파일에 정의되어 있다면, 바로 사용!
env = CoolingEnv()

# 환경 체크 (버그 사전 탐지)
check_env(env, warn=True)

# VecEnv로 감싸주기 (SB3는 VecEnv 요구)
vec_env = DummyVecEnv([lambda: env])

class EpisodeRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_temp_penalties = []
        self.episode_energy_penalties = []
        self.episode_near_overheat_penalties = []

    def _on_step(self) -> bool:
        # Gymnasium 벡터 환경이면 infos는 리스트
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        for info, done in zip(infos, dones):
            if done:
                # 에피소드 종료시에만 누적 페널티 저장
                if (
                    "total_temp_penalty" in info and
                    "total_energy_penalty" in info and
                    "total_near_overheat_penalty" in info
                ):
                    self.episode_temp_penalties.append(info["total_temp_penalty"])
                    self.episode_energy_penalties.append(info["total_energy_penalty"])
                    self.episode_near_overheat_penalties.append(info["total_near_overheat_penalty"])
        return True

# DQN 모델 생성
model = DQN(
    "MlpPolicy",           
    env, 
    learning_rate=1e-3 ,
    buffer_size=10000,
    batch_size=64,
    train_freq=4,
    target_update_interval=500,
    gamma=0.99,
    exploration_fraction=0.8,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.02,
    verbose=1
)

# 학습 (타임스텝 10만은 예시, 필요시 조절)
callback = EpisodeRewardCallback()
model.learn(total_timesteps=100000, callback=callback)

# 모델 저장
model.save("cooling_dqn_model_2")


episodes = range(1, len(callback.episode_temp_penalties) + 1)
total_penalties = (
    np.array(callback.episode_temp_penalties) +
    np.array(callback.episode_energy_penalties) +
    np.array(callback.episode_near_overheat_penalties)
)
fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

axes[0].plot(episodes, total_penalties, color='black')
axes[0].set_ylabel("all penalty")
axes[0].set_title("Total penalty per episode")

axes[1].plot(episodes, callback.episode_temp_penalties, color='blue')
axes[1].set_ylabel("temp penalty")
axes[1].set_title("Temp penalty per episode")

axes[2].plot(episodes, callback.episode_energy_penalties, color='orange')
axes[2].set_ylabel("energy penalty")
axes[2].set_title("Energy penalty per episode")

axes[3].plot(episodes, callback.episode_near_overheat_penalties, color='green')
axes[3].set_ylabel("near overheat penalty")
axes[3].set_xlabel("episodes")
axes[3].set_title("Near overheat penalty per episode")

plt.tight_layout()
plt.show()

# 테스트: 학습된 정책 시뮬레이션
obs = env.reset()[0]
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated