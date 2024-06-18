import os
from omegaconf import OmegaConf

from omnigibson import gm
from omnigibson.utils.transform_utils import l2_distance
from ..env import OmnigibEnv

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True


class CubeEnv(OmnigibEnv):
    def __init__(self):

        base_cfg = OmegaConf.load(os.path.join(__file__, "configs/base.yaml"))
        cube_cfg = OmegaConf.load(os.path.join(__file__, "configs/cube.yaml"))
        cfg = OmegaConf.merge(base_cfg, cube_cfg)

        super().__init__(cfg)
        self.cube = self.og_env.scene.object_registry("name", "cube")

    def internal_obs(self, raw_frame):
        state = self.render()
        robot_pos = self.robot.get_position()
        cube_pos = self.cube.get_position()
        dist = l2_distance(robot_pos, cube_pos)
        reward = -dist
        done = False
        info = {"testing": "hi"}
        state = self.render()
        # assert np.isclose(self.vision_sensor.get_position(), [0, 0, 1]).all()
        # assert np.isclose(self.vision_sensor.get_orientation(), [0.70721355, 0., 0., 0.707]).all()
        return state, reward, done, info


# state = self.render()
# robot_pos = self.robot.get_position()
# cube_pos = self.cube.get_position()
# # Sparse reward is received if distance between robot_idn robot's eef and goal is below the distance threshold
# # success = l2_distance(robot_pos, cube_pos) < 0.1
# reward = -l2_distance(robot_pos, cube_pos)
# done = False
# info = {"testing": "hi"}
# return state, reward, done, info


def make_env(cfg_dict):
    print("Constructing environment with parameters:", cfg_dict)
    return CubeEnv()
