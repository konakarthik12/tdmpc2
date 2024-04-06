import gym
import mani_skill2.envs
from mani_skill2 import PACKAGE_ASSET_DIR
import urchin
from Transform2Act.xml_robot import Body, Joint, Robot, Actuator
env = gym.make(
    'PickSingleYCB-v0',
    obs_mode='state',
    control_mode='pd_ee_delta_pos',
    render_camera_cfgs=dict(width=384, height=384),
)

urdf_file_path = env.env.agent.urdf_path.format(PACKAGE_ASSET_DIR=PACKAGE_ASSET_DIR)
print(urdf_file_path)



from urchin import URDF

robot = URDF.load(urdf_file_path)

children = {}
parent = {}
for link in robot.links:
    children[link.name] = []
    parent[link.name] = None
for joint in robot.joints:
    children[joint.parent].append(joint.child)
    parent[joint.child] = joint.parent


def strip_prefix(string):
    if string.startswith("panda_"):
        string = string[6:]
    return string
for link in robot.links:
    print(strip_prefix(link.name))
for joint in robot.joints:
    print('{} {} {}'.format(
        strip_prefix(joint.child), strip_prefix(joint.parent), strip_prefix(joint.name)
    ))

robot = env.env.agent.robot
for link in robot.get_links():
    print(link, "link pose", link.pose, "inertia", link.inertia)

for joint in robot.get_joints():
    print(joint, joint.articulation.pose, joint.type)

robot = Robot()