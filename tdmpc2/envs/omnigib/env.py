import gym
import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

import omnigibson as og
from omnigibson.sensors import VisionSensor


def fix_angle(angle):
    return np.roll(angle, -1)


def to_omni_quat(euler_angle):
    # return (Rotation.from_quat(fix_angle([-0.15475,0.01625,0.27298,0.94935])).as_euler('xyz',degrees=True))

    return np.roll(Rotation.from_euler("xyz", euler_angle, degrees=True).as_quat(), 1)


class OmnigibEnv(gym.Env):

    def __init__(self, omni_cfg):
        self.cfg = omni_cfg
        print("Creating the environment with config: ", self.cfg)
        self.og_env = og.Environment(configs=self.cfg)
        self.robot = self.og_env.robots[0]
        self.vision_sensor = None

        for sensor in self.robot.sensors.values():
            if isinstance(sensor, VisionSensor):
                self.vision_sensor = sensor
        # sensor.image_height = 64
        # sensor.image_width = 64
        print("huh", self.og_env.external_sensors.keys())

        # TODO: hard code name of vision sensor
        external_camera: VisionSensor = list(self.og_env.external_sensors.values())[0]
        self.vision_sensor = external_camera

        # self.vision_sensor = self.og_env.external_sensors["external_camera"]

        # assert self.vision_sensor.image_height == 64 and self.vision_sensor.image_width == 64
        print("Vision sensor found with image shape: ", self.vision_sensor.image_height, self.vision_sensor.image_width)

        # # Update the simulator's viewer camera's pose so it points towards the robot
        # self.vision_sensor.set_position_orientation(
        # 	position=np.array([0.7846293172674343, 1.1133845211995728, 2.4648582197875495]),
        # 	orientation=fix_angle([-0.15475, 0.01625, 0.27298, 0.94935])
        #
        # 	# orientation (-53.1, -46.17, -151.588)
        # 	# orientation=Rotation.from_euler("xyz", [65.43011, -0.000413565464, -127.512396], degrees=True).as_quat()
        #
        # )

        self.max_episode_steps = 1000

        rgb_image = self.render()
        print("RGB Image shape: ", rgb_image.shape)
        # assert rgb_image.shape == (64, 64, 3)
        print("RGB Image shape: ", rgb_image.shape)

        self.observation_space = Box(0, 255, rgb_image.shape, np.uint8)

        print("Observation space: ", self.observation_space)
        self.action_space = self.og_env.action_space["robot0"]
        print("Action space: ", self.action_space)



    def render(self, mode="rgb_array", width=None, height=None):
        assert mode == "rgb_array", "Only rgb_array mode is supported"
        # assert width == 64 and height == 64, "Only 64x64 resolution is supported"
        # render the environment
        # print("Rendering the environment")
        camera_data, unknown = self.vision_sensor.get_obs()
        rgb_image = camera_data["rgb"]
        (actual_height, actual_width, _) = rgb_image.shape
        if width is not None or height is not None:
            assert width == actual_width and height == actual_height, f"Requested resolution ({width}x{height}) does not match actual resolution ({actual_width}x{actual_height})"
        # assert rgb_image.shape == (64, 64, 4)
        # remove the alpha channel, changes shape from (64, 64, 4) to (64, 64, 3)
        rgb_image = rgb_image[:, :, :3]
        # assert rgb_image.shape == (64, 64, 3)
        return rgb_image

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        # reset the environment
        print("Resetting the environment")
        raw_frame = self.og_env.reset()

        state, reward, done, info = self.internal_obs(raw_frame)
        # assert state.shape == (64, 64, 3), "State shape is not (64, 64, 3), it is: " + str(state.shape)
        # assert state in self.observation_space
        # return state, info
        return state

    def step(self, action):
        raw_frame = self.og_env.step(action)
        state, reward, done, info = self.internal_obs(raw_frame)
        terminated = done
        truncated = False
        return state, reward, terminated, truncated, info

    def close(self):
        # close the environment
        print("Closing the environment")
        self.og_env.close()

    def internal_obs(self, raw_frame):
        rgb_image = self.render()
        state = rgb_image
        reward = raw_frame[1]
        done = False
        info = {"testing": "hi"}
        self._last_info = info
        return state, reward, done, info
# return state

#
# #environment that is based on Google's DMControl suite
# class DMControlEnv(gym.Env):
