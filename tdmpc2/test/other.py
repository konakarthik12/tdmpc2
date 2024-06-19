import numpy as np
# from gymnasium.spaces import Dict, Box
#
# observation_space = Dict({"robot0":
#                                    Box(0, 255, (128, 128, 4), dtype=np.uint8),
#                                })
# print("Observation space: ", observation_space)
# sample = observation_space.sample()
# print(sample)

import os
# import mujoco_py
# mj_path = mujoco_py.utils.discover_mujoco()
# xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
# model = mujoco_py.load_model_from_path(xml_path)
# sim = mujoco_py.MjSim(model)
#
# print(sim.data.qpos)
#
# sim.step()
# print(sim.data.qpos)


import torch
assert torch.cuda.is_available()



import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="tdmpc2",
    entity="greenpizza12",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()