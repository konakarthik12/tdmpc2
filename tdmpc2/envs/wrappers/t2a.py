from collections import deque

import gym
import numpy as np
import torch


class T2AWrapper(gym.Wrapper):
	"""
	Wrapper for Transform2Act environments.
    In these envs, the observation is image + dict state.
	"""

	def __init__(self, cfg, env, num_frames=3, render_size=64):
		super().__init__(env)
		self.cfg = cfg
		self.env = env
  
		# Update observation space
		self.observation_space = gym.spaces.Dict()
  
		# image
		self.observation_space.spaces['rgb'] = gym.spaces.Box(
            low=0, high=255, shape=(num_frames*3, render_size, render_size), dtype=np.uint8
        )
  
		# dict state
		sample_obs = self.env._get_obs()
		node_obs = sample_obs[0]	
		edge_obs = sample_obs[1]
		num_nodes = node_obs.shape[0]
  
		self.observation_space.spaces['node'] = gym.spaces.Box(
			low=-1., high=1., shape=node_obs.shape, dtype=np.float32
		)
		self.observation_space.spaces['edge'] = gym.spaces.Box(
			low=0, high=num_nodes, shape=edge_obs.shape, dtype=np.int32
		)
		self._frames = deque([], maxlen=num_frames)
		self._render_size = render_size

	def _get_obs(self):
		# image
		frame = self.env.render(
			mode='rgb_array', width=self._render_size, height=self._render_size
		).transpose(2, 0, 1)
		self._frames.append(frame)
		img_obs = torch.from_numpy(np.concatenate(self._frames))
  
		# dict state
		state = self.env._get_obs()
		node = state[0]
		edge = state[1]
		node_obs = torch.from_numpy(node).to(dtype=torch.float32)
		edge_obs = torch.from_numpy(edge).to(dtype=torch.long)
  
		return {'rgb': img_obs, 'node': node_obs, 'edge': edge_obs}

	def _get_graph_obs(self):
		# dict state
		state = self.env._get_obs()
		node = state[0]
		edge = state[1]
		node_obs = torch.from_numpy(node).to(dtype=torch.float32)
		edge_obs = torch.from_numpy(edge).to(dtype=torch.long)
  
		return {'node': node_obs, 'edge': edge_obs}

	def _get_node_obs_size(self):
		return self.observation_space.spaces['node'].shape[1]

	def reset(self):
		self.env.reset()
		for _ in range(self._frames.maxlen):
			obs = self._get_obs()
		return obs

	def step(self, action):
		if action.ndim == 1:
			action = action.reshape((-1, 1))
		if isinstance(action, torch.Tensor):
			action = action.cpu().numpy()
		_, reward, done, info = self.env.step(action)
		reward = torch.tensor(reward, dtype=torch.float32)
		return self._get_obs(), reward, done, info

	def rand_act(self):
		return torch.from_numpy(self.env.action_space.sample()).to(dtype=torch.float32)