import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np

import hydra
from termcolor import colored

from common.parser import parse_cfg
from envs import make_env

from collect.ppo import CollectPPO
from collect.sac import CollectSAC
from envs.wrappers.tensor import TensorWrapper
import gym

torch.backends.cudnn.benchmark = True

'''
Gym class that SB3 can use
'''
class SB3Env(gym.Env):
	
	def __init__(self, env):
		super().__init__()
  
		self.env = env
		# remove the TensorWrapper
		if isinstance(self.env, TensorWrapper):
			self.env = env.env
		
		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space
		self.render_mode = "rgb_array"
		
	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		truncated = done	# SB3 requires this
		return obs, reward, done, truncated, info

	def reset(self, seed=None, options=None):
		obs = self.env.reset()
		info = {}
		return obs, info

	def render(self):
		return self.env.render()

	def close(self):
		return self.env.close()


@hydra.main(config_name='config', config_path='.')
def collect(cfg: dict):
	"""
	Script for collecting trajectories using SB3 agents.
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	logdir = './'
	logdir = os.path.abspath(logdir)
	print(colored('Work dir:', 'yellow', attrs=['bold']), logdir)
	
	cfg = parse_cfg(cfg)

	env = make_env(cfg)
	env = SB3Env(env)
 
	if cfg.sb3_algo == "ppo":
		model = CollectPPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
	elif cfg.sb3_algo == "sac":
		model = CollectSAC('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
	else:
		raise ValueError(f"Unsupported SB3 algorithm: {cfg.sb3_algo}")
	model.learn(total_timesteps=cfg.steps,)
	print('\nTraining completed successfully')
 
	'''
	How to parse saved trajectory
 	'''
	traj_path = os.path.join(logdir, 'Traj', "000000.zip")
	if os.path.exists(traj_path):
		print(f"Loading trajectory from {traj_path}")

		# unzip the file
		import zipfile
		with zipfile.ZipFile(traj_path, 'r') as zip_ref:
			unzip_path = os.path.join(logdir, 'Traj', "000000")
			zip_ref.extractall(unzip_path)

		traj_path = unzip_path
		# load the trajectory
		prev_obs = np.load(os.path.join(traj_path, 'prev_obs.npy'))
		next_obs = np.load(os.path.join(traj_path, 'next_obs.npy'))
		prev_rgb = np.load(os.path.join(traj_path, 'prev_rgb.npy'))
		next_rgb = np.load(os.path.join(traj_path, 'next_rgb.npy'))
		action = np.load(os.path.join(traj_path, 'action.npy'))
		reward = np.load(os.path.join(traj_path, 'reward.npy'))
  
		# save first 10 frames
		num_frames = min(10, len(prev_rgb))
		for i in range(num_frames):
			step_path = os.path.join(traj_path, f'step_{i}')
			os.makedirs(step_path, exist_ok=True)
   
			# save the images
			prev_rgb_path = os.path.join(step_path, 'prev_rgb.png')
			next_rgb_path = os.path.join(step_path, 'next_rgb.png')
			from PIL import Image
			Image.fromarray(prev_rgb[i]).save(prev_rgb_path)
			Image.fromarray(next_rgb[i]).save(next_rgb_path)
   
			# save the observation as txt
			prev_obs_path = os.path.join(step_path, 'prev_obs.txt')
			next_obs_path = os.path.join(step_path, 'next_obs.txt')
			np.savetxt(prev_obs_path, prev_obs[i])
			np.savetxt(next_obs_path, next_obs[i])
   
			# save the action and reward
			action_path = os.path.join(step_path, 'action.txt')
			reward_path = os.path.join(step_path, 'reward.txt')
			np.savetxt(action_path, action[i])
			np.savetxt(reward_path, reward.reshape(-1, 1)[i])

if __name__ == '__main__':
	collect()
