from collections import deque

import gym
import numpy as np
import torch

# MJCF parser
from envs.wrappers.mjgraph.mjcf_parser import Robot


class MJGraphWrapper(gym.Wrapper):
	"""
	Wrapper for environments that use MuJoCo.
	In these envs, the observation is image + dict state.
	"""

	def __init__(self, model, env, num_frames=3, render_size=96):
		super().__init__(env)
  
		# env
		self.env = env
  
		# Update observation space
		self.observation_space = gym.spaces.Dict()
  
		# state
		self.observation_space.spaces['state'] = env.observation_space
  
		# image
		self.observation_space.spaces['srgb'] = gym.spaces.Box(
			low=0, high=255, shape=(num_frames*3, render_size, render_size), dtype=np.uint8
		)
  
		# MJCF parser
		self.model = model
		self.mjcf_robot = Robot(model)
  
		# initialize node and edge observation
		# for now, these observations are fixed
		self.init_node_obs()
		self.init_edge_obs()
  
		# dict state: node and edge, retrieved from robot
		self.observation_space.spaces['node'] = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=self.node_obs.shape, dtype=np.float32
		)
		self.observation_space.spaces['edge'] = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=self.edge_obs.shape, dtype=np.float32
		)
		self._frames = deque([], maxlen=num_frames)
		self._render_size = render_size
  
	def init_node_obs(self):
		'''
		Node represents a body in the robot, and it contains the following information:

		- pos: body pos (3)
  
		* geom: at most 4 geoms per body
		- type: 01 (capsule), 10 (sphere)
		- size: float (3)
		- pos: float (3)
		- fromto: float (6)
  		'''
	
		self.node_idx = {}
		self.node_obs = []
		self.node_num_geom = []		# number of geoms per node
	
		node_id = 0
		for body in self.mjcf_robot.bodies:
	  
			# if there is no geom, skip
			if len(body.geoms) == 0:
				continue

			if len(body.geoms) > 4:
				raise ValueError(f'Too many geoms: {len(body.geoms)}')

			geom_feats = np.zeros((4, 17), dtype=np.float32)
			try:
				for gi, geom in enumerate(body.geoms):
					geom_feat = np.zeros((17,), dtype=np.float32)
					# type
					if geom.type == 'capsule':
						geom_feat[:2] = 0, 1
					elif geom.type == 'sphere':
						geom_feat[:2] = 1, 0
					else:
						raise ValueError(f'Unknown geom type: {geom.type}')
					# size
					if geom.size.shape[0] == 1:
						geom_feat[2] = geom.size
					elif geom.size.shape[0] == 2:
						geom_feat[2:4] = geom.size
					elif geom.size.shape[0] == 3:
						geom_feat[2:5] = geom.size
					else:
						raise ValueError(f'Unknown geom size shape: {geom.size.shape}')
					# pos
					geom_feat[5:8] = geom.pos
	
     				# fromto
					geom_feat[8:14] = geom.fromto
     
					# body pos
					geom_feat[14:17] = body.pos
	 
					geom_feats[gi] = geom_feat
  
			except Exception as e:
				print(f'Error in init_node_obs: {e}')
				exit(-1)
    
			self.node_obs.append(geom_feats)
			self.node_num_geom.append(len(body.geoms))
   
			self.node_idx[body.name] = node_id
			node_id += 1
   
		self.node_obs = np.array(self.node_obs, dtype=np.float32)
		self.node_num_geom = np.array(self.node_num_geom, dtype=np.int32)
  
	def init_edge_obs(self):
		'''
		Edge represents a connection between two bodies in the robot, and it contains the following information:
  
		* joint: at most 4 joints per body
		- parent node id (1)
		- child node id (1)
		- type (2)
		- range (2)
		- position (3)
		- axis (3)
  		'''
    
		self.edge_obs = []
		self.edge_num_joint = []		# number of joints per edge
  
		for body in self.mjcf_robot.bodies:
	  
			# if there is no joint, skip
			if len(body.joints) == 0:
				continue

			# if there is no parent node, skip
			if body.parent is None:
				continue

			if len(body.joints) > 4:
				raise ValueError(f'Too many joints: {len(body.joints)}')

			joint_feats = np.zeros((4, 12), dtype=np.float32)
			try:
				for ji, joint in enumerate(body.joints):
					joint_feat = np.zeros((12,), dtype=np.float32)
					# parent node id
					joint_feat[0] = self.node_idx[body.parent.name]
					# child node id
					joint_feat[1] = self.node_idx[body.name]
					# type
					if joint.type == 'hinge':
						joint_feat[2:4] = 0, 1
					elif joint.type == 'slide':
						joint_feat[2:4] = 1, 0
					else:
						raise ValueError(f'Unknown joint type: {joint.type}')
					# range
					joint_feat[4:6] = joint.range
					# position
					joint_feat[6:9] = joint.pos
					# axis
					joint_feat[9:12] = joint.axis
  
					joint_feats[ji] = joint_feat
  
			except Exception as e:
				print(f'Error in init_edge_obs: {e}')
				exit(-1)
	
			self.edge_obs.append(joint_feats)
			self.edge_num_joint.append(len(body.joints))
   
		self.edge_obs = np.array(self.edge_obs, dtype=np.float32)
		self.edge_num_joint = np.array(self.edge_num_joint, dtype=np.int32)
  
	def render(self, mode: str='rgb_array'):
		return self.env.render(mode, width=self._render_size, height=self._render_size)

	def _get_obs(self):
		# image
		frame = self.render('rgb_array').transpose(2, 0, 1)
		self._frames.append(frame)
		img_obs = torch.from_numpy(np.concatenate(self._frames))
  
		# dict state
		node_obs = self.node_obs
		edge_obs = self.edge_obs
  
		return {'srgb': img_obs, 'node': node_obs, 'edge': edge_obs}

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
		state = self.env.reset()
		for _ in range(self._frames.maxlen):
			obs = self._get_obs()
		obs['state'] = state
		return obs

	def step(self, action):
		# if action.ndim == 1:
		# 	action = action.reshape((-1, 1))
		if isinstance(action, torch.Tensor):
			action = action.cpu().numpy()
		state, reward, done, info = self.env.step(action)
		
		obs = self._get_obs()
		obs['state'] = state
		reward = torch.tensor(reward, dtype=torch.float32)
		return obs, reward, done, info

	def rand_act(self):
		return torch.from_numpy(self.env.action_space.sample()).to(dtype=torch.float32)