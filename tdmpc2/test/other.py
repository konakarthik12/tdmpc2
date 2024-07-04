import os
#
import yaml
from pprint import pprint
from omegaconf import OmegaConf
import json
#
import omnigibson as og
from omnigibson.macros import gm
# from omnigibson.utils.ui_utils import choose_from_options
from tdmpc2.envs.omnigib.env import OmnigibEnv
from tdmpc2.envs.omnigib.tasks.cube import CubeEnv

#
# # Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False

# print("started")
# # Load the pre-selected configuration and set the online_sampling flag
config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
# print(json.dumps(cfg, indent=4))
print()
alt = OmegaConf.load("/home/kkona/Documents/research/tdmpc2/tdmpc2/envs/omnigib/tasks/configs/base.yaml")
alt = OmegaConf.to_container(alt)
alt = dict(alt)
# print(json.dumps(alt, indent=4))
assert cfg == alt

# # cfg["task"]["online_object_sampling"] = should_sample
#
# # Load the environment
env = CubeEnv().og_env
#
# # Allow user to move camera more easily
# og.sim.enable_viewer_camera_teleoperation()
#
# # Run a simple loop and reset periodically
for j in range(2):
    og.log.info("Resetting environment")
    env.reset()
    for i in range(100):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            og.log.info("Episode finished after {} timesteps".format(i + 1))
            break
#
# # Always close the environment at the end
# env.close()
