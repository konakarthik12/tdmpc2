import numpy as np
import os
import shutil

class Collector:
    
    def __init__(self, logdir: str):
        
        self.logdir = logdir
        
        self._acc_reward = 0.0
        self._max_acc_reward = -np.inf
        
        self._traj_cnt = 0
        self._traj_prev_obs = []
        self._traj_prev_rgb = []
        self._traj_action = []
        self._traj_reward = []
        self._traj_next_obs = []
        self._traj_next_rgb = []
        
    def add_step(self, prev_obs, prev_rgb, action, reward, next_obs, next_rgb, done):
        
        self._acc_reward += reward
        
        self._traj_prev_obs.append(prev_obs)
        self._traj_prev_rgb.append(prev_rgb)
        self._traj_action.append(action)
        self._traj_reward.append(reward)
        self._traj_next_obs.append(next_obs)
        self._traj_next_rgb.append(next_rgb)
        
        if done:
            if self._acc_reward > self._max_acc_reward:
                self._max_acc_reward = self._acc_reward
                
                traj_dir = os.path.join(self.logdir, f'{self._traj_cnt:06d}')
                os.makedirs(traj_dir, exist_ok=True)
                
                # exclude last element, because it could be the terminal state
                prev_obs = np.array(self._traj_prev_obs[:-1])
                prev_rgb = np.array(self._traj_prev_rgb[:-1])
                action = np.array(self._traj_action[:-1])
                reward = np.array(self._traj_reward[:-1])
                next_obs = np.array(self._traj_next_obs[:-1])
                next_rgb = np.array(self._traj_next_rgb[:-1])
                
                np.save(os.path.join(traj_dir, 'prev_obs.npy'), prev_obs)
                np.save(os.path.join(traj_dir, 'prev_rgb.npy'), prev_rgb)
                np.save(os.path.join(traj_dir, 'action.npy'), action)
                np.save(os.path.join(traj_dir, 'reward.npy'), reward)
                np.save(os.path.join(traj_dir, 'next_obs.npy'), next_obs)
                np.save(os.path.join(traj_dir, 'next_rgb.npy'), next_rgb)
                
                # zip the files
                shutil.make_archive(traj_dir, 'zip', traj_dir)
                print(f"===== Saved trajectory at {traj_dir}.zip")
                
                # remove the directory
                shutil.rmtree(traj_dir)
                
                self._traj_cnt += 1
                
                # Reset trajectory
                self._traj_prev_obs = []
                self._traj_prev_rgb = []
                self._traj_action = []
                self._traj_reward = []
                self._traj_next_obs = []
                self._traj_next_rgb = []
                
                self._acc_reward = 0.0