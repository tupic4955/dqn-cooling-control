import matplotlib.pyplot as plt
import numpy as np
from CoolingEnv import CoolingEnv
# 모델, 환경 불러오기
from stable_baselines3 import DQN

# (환경 코드가 같은 파일에 있다면 아래 코드로 환경 생성)
env = CoolingEnv()
model = DQN.load("cooling_dqn_model")

obs, _ = env.reset()
done = False

# 온도, 유량, 오버히트 셀 로그
temps_log = []
flows_log = []
num_overheat_log = []


while not done:
    # DQN 정책으로 액션 선택
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 로그 저장
    temps_log.append(env.sim.get_state().copy())
    flows_log.append(env.flows.copy())
    #num_overheat_log.append(info["num_overheated"])
    
    # (옵션) 중간 중간 실시간 플롯
    # env.render()
    done = terminated or truncated

temps_log = np.array(temps_log)        # (스텝, 3, 3)
flows_log = np.array(flows_log)        # (스텝, 4)
num_overheat_log = np.array(num_overheat_log)

# ---------- 시각화 ----------
steps = np.arange(len(temps_log))

# 1. 전체 스텝별 평균/최대/최소 온도 플롯
plt.figure(figsize=(10,5))
plt.plot(steps, temps_log.reshape(len(temps_log), -1).mean(axis=1), label="temp_average")
plt.plot(steps, temps_log.reshape(len(temps_log), -1).max(axis=1), label="temp_max")
plt.plot(steps, temps_log.reshape(len(temps_log), -1).min(axis=1), label="temp_min")
plt.xlabel("Step")
plt.ylabel("temp(°C)")
plt.legend()
plt.title("Cell temperature changes while the agent is running")
plt.grid()
plt.show()

# 2. 각 파이프별 유량 변화 플롯
plt.figure(figsize=(10,5))
for i in range(4):
    plt.plot(steps, flows_log[:, i], label=f"pipe {i} flow rate")
plt.xlabel("Step")
plt.ylabel("flow rate")
plt.legend()
plt.title("Flow rate change by pipe")
plt.grid()
plt.show()

# 3. (옵션) 각 step에서 온도 heatmap 애니메이션
from matplotlib import animation

fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(temps_log[0], vmin=15, vmax=np.max(temps_log))
cbar = fig.colorbar(im, ax=ax)

def update_heatmap(frame):
    im.set_data(temps_log[frame])
    ax.set_title(f"Step {frame} (Overheat cells: {num_overheat_log[frame]})")
    # colorbar는 재생성하지 않음!
    return [im]

ani = animation.FuncAnimation(fig, update_heatmap, frames=len(temps_log), interval=200, blit=True)
plt.show()