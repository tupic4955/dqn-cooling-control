import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN

# 환경 불러오기
from CoolingEnv import CoolingEnv  # 당신이 정의한 환경 파일로 대체
from gymnasium.envs.registration import register

# 환경 등록 (필요한 경우)
register(
    id="CoolingEnv-v0",
    entry_point="CoolingEnv:CoolingEnv"
)

# 환경 생성
env = gym.make("CoolingEnv-v0")
obs, _ = env.reset()

# 학습된 모델 불러오기
model = DQN.load("cooling_dqn_agent")

# 시각화 준비
plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((3, 3)), cmap="hot", vmin=15, vmax=100)
cbar = fig.colorbar(im, ax=ax)

for t in range(300):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    temps = env.unwrapped.sim.get_state()
    
    im.set_data(temps)
    ax.set_title(f"Step {t+1}, Reward: {reward:.2f}")
    plt.pause(0.05)

plt.ioff()
plt.show()