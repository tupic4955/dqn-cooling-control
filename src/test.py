from cooling_model import CoolingSimulation

sim = CoolingSimulation()
flows = [0.5, 0.8, 0.2, 1.0]

for t in range(10):
    sim.step(flows)
    print(sim.get_state())