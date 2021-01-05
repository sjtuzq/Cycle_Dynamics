import mujoco_py
import numpy as np

fullpath = './xarm7_mocap.xml'
model = mujoco_py.load_model_from_path(fullpath)
sim = mujoco_py.MjSim(model, nsubsteps=3)
viewer = mujoco_py.MjViewer(sim)

start = np.array([0.0,0.314,0.198])
target = np.array([0.25,0.60,0.6])
step_n = 2000
step_size = 0.0005
for i in range(step_n):
        sim.data.mocap_pos[:] = sim.data.mocap_pos + (target-start)*step_size
        sim.step()
        viewer.render()
        print(i,sim.data.mocap_pos)

