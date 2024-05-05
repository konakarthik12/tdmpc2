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
from envs.wrappers.t2a import T2AWrapper
from envs.wrappers.mjgraph.mjgraph import MJGraphWrapper
import gym

from copy import deepcopy

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
  
		'''
		If it is a t2a environment, observation is comprised of:
		- rgb
		- node
		- edge
		
		We only use rgb image for trajectory collection.
  		'''
		self.is_t2a_env = isinstance(self.env, T2AWrapper)
  
		'''
		If it is a mjgraph environment, observation is comprised of:
		- state
		- rgb
		- node
		- edge
  		'''
		self.is_mjgraph_env = isinstance(self.env, MJGraphWrapper)
		if self.is_mjgraph_env:
			self.observation_space = self.env.observation_space.spaces['state']
   
		self.render_mode = "rgb_array"
		self._last_info = None
		
	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		truncated = done	# SB3 requires this
		
		if self.is_mjgraph_env:
			info['srgb'] = obs['srgb']
			info['node'] = obs['node']
			info['edge'] = obs['edge']
			obs = obs['state']
			self._last_info = deepcopy(info)
   
		return obs, reward, done, truncated, info

	def reset(self, seed=None, options=None):
		obs = self.env.reset()
		info = {}

		if self.is_mjgraph_env:
			info['srgb'] = obs['srgb']
			info['node'] = obs['node']
			info['edge'] = obs['edge']
			obs = obs['state']
			self._last_info = deepcopy(info)
   
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
	# if env is MJGraphWrapper, save model
	if isinstance(env, MJGraphWrapper):
		model_path = os.path.join(logdir, 'model.xml')
		with open(model_path, 'w') as f:
			model_str = env.model.decode('utf-8')
			f.write(model_str)
	env = SB3Env(env)
 
	'''
	If the environment is T2A environment, we use CNN policy.
	Otherwise, we use MLP policy.
 	'''
	policy = 'MultiInputPolicy' if (env.is_t2a_env) else 'MlpPolicy'
 
	if cfg.sb3_algo == "ppo":
		model = CollectPPO(policy, env, verbose=1, tensorboard_log=logdir)
	elif cfg.sb3_algo == "sac":
		model = CollectSAC(policy, env, verbose=1, tensorboard_log=logdir)
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
		if env.is_t2a_env or env.is_mjgraph_env:
			# load the trajectory
			prev_node = np.load(os.path.join(traj_path, 'prev_obs_node.npy'))
			next_node = np.load(os.path.join(traj_path, 'next_obs_node.npy'))
			prev_edge = np.load(os.path.join(traj_path, 'prev_obs_edge.npy'))
			next_edge = np.load(os.path.join(traj_path, 'next_obs_edge.npy'))
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
	
				# save the node as txt
				prev_node_path = os.path.join(step_path, 'prev_node.txt')
				next_node_path = os.path.join(step_path, 'next_node.txt')
				np.savetxt(prev_node_path, prev_node[i, 0])
				np.savetxt(next_node_path, next_node[i, 0])
    
				# save the edge as txt
				prev_edge_path = os.path.join(step_path, 'prev_edge.txt')
				next_edge_path = os.path.join(step_path, 'next_edge.txt')
				np.savetxt(prev_edge_path, prev_edge[i, 0])
				np.savetxt(next_edge_path, next_edge[i, 0])
	
				# save the action and reward
				action_path = os.path.join(step_path, 'action.txt')
				reward_path = os.path.join(step_path, 'reward.txt')
				np.savetxt(action_path, action[i])
				np.savetxt(reward_path, reward.reshape(-1, 1)[i])
      
		else:
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
