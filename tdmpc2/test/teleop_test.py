"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import IsGrasping, Touching
from omnigibson.robots import Fetch
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController
from omegaconf import OmegaConf
from hydra import compose, initialize

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True


def main():
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    with initialize(version_base=None, config_path="../envs/omnigib/tasks/configs"):
        cfg = compose(config_name="cube")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # cfg["robots"] = [robot0_cfg]
    # Create the environment
    env = og.Environment(configs=cfg)
    robot: Fetch = env.robots[0]

    cube = env.scene.object_registry("name", "cube")
    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([1.46949, -3.97358, 2.21529]),
        orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )

    for i in range(100):
        action = np.zeros(robot.action_space.shape)

        env.step(action=action)

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)
    #
    # # Register custom binding to reset the environment
    # action_generator.register_custom_keymapping(
    #     key=lazy.carb.input.KeyboardInput.R,
    #     description="Reset the robot",
    #     callback_fn=lambda: env.reset(),
    # )
    #
    # # # Print out relevant keyboard info if using keyboard teleop
    action_generator.print_keyboard_teleop_info()
    #
    # # Other helpful user info
    # print("Running demo.")
    # print("Press ESC to quit")

    # robot is facing +x
    # action[4] is forward (+x)
    # action[5] is left (+y)
    # action[6] is up (+z)
    # action[7] is rotate x-axis
    # action[8] is rotate y-axis
    # action[9] is rotate z-axis
    # action[10] is gripper (1 is open, -1 is close)

    # Reset environment and robot
    env.reset()
    robot.reset()
    step = 0
    while step != -1:
        action = action_generator.get_teleop_action()
        action = action[4:]
        # print(robot.states.keys())
        # print(robot.states[Touching].get_value(cube)), print(robot.states[IsGrasping].get_value(cube))
        env.step(action=action)
        step += 1

    # Always shut down the environment cleanly at the end
    env.close()


if __name__ == "__main__":
    main()
