import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class CoolingSimulation:
    def __init__(self):
        self.grid_shape = (3, 3)
        self.heat_gen = 1.0
        self.cooling_efficiency = 0.5
        self.init_temp = 20.0
        self.pipe_count = 4
        self.max_flow = 1.0
        self.temps = np.full(self.grid_shape, self.init_temp)

        self.pipes = {
            0: [((0, 0), (0, 1)), ((1, 0), (1, 1)), ((2, 0), (2, 1))],
            1: [((0, 1), (0, 2)), ((1, 1), (1, 2)), ((2, 1), (2, 2))],
            2: [((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2))],
            3: [((1, 0), (2, 0)), ((1, 1), (2, 1)), ((1, 2), (2, 2))]
        }

    def step(self, flows):
        self.temps += self.heat_gen
        for pipe_id in range(self.pipe_count):
            flow = np.clip(flows[pipe_id], 0, self.max_flow)
            coolant_temp = 15.0
            for cell_from, cell_to in self.pipes[pipe_id]:
                avg_temp = (self.temps[cell_from] + self.temps[cell_to]) / 2
                heat_absorbed = flow * self.cooling_efficiency * max(avg_temp - coolant_temp, 0)
                self.temps[cell_from] -= heat_absorbed / 2
                self.temps[cell_to] -= heat_absorbed / 2
                coolant_temp += heat_absorbed

    def get_state(self):
        return self.temps.copy()

    def reset(self):
        self.temps = np.full(self.grid_shape, self.init_temp)


# 시뮬레이터 실행
sim = CoolingSimulation()
flows = [0.5, 0.5, 0.5, 0.5]

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.35)
im = ax.imshow(sim.get_state(), cmap='coolwarm', vmin=15, vmax=40)
text_matrix = [[ax.text(j, i, f"{sim.temps[i, j]:.1f}", ha='center', va='center') for j in range(3)] for i in range(3)]
ax.set_title("Cooling Simulation")
plt.colorbar(im, ax=ax)

# 슬라이더 위치 설정
axcolor = 'lightgoldenrodyellow'
sliders = []
slider_axes = []
for i in range(4):
    ax_slider = plt.axes([0.25, 0.25 - i*0.05, 0.65, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, f'Flow {i}', 0.0, 1.0, valinit=flows[i])
    sliders.append(slider)
    slider_axes.append(ax_slider)

# 버튼 (1스텝 진행)
step_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(step_ax, 'Step', color='lightblue', hovercolor='skyblue')

def update(val):
    for i in range(4):
        flows[i] = sliders[i].val

    sim.step(flows)
    new_state = sim.get_state()
    im.set_data(new_state)

    for i in range(3):
        for j in range(3):
            text_matrix[i][j].set_text(f"{new_state[i, j]:.1f}")

    fig.canvas.draw_idle()

button.on_clicked(update)
plt.show()

