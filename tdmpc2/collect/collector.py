import numpy as np
import os
import shutil

class Collector:
    
    def __init__(self, logdir: str):
        
        self.logdir = logdir
        
        self._acc_reward = 0.0
        self._max_acc_reward = -np.inf
        
        self._traj_cnt = 0
        
        self._traj_prev_obs = {}        # observation
        self._traj_prev_rgb = []        # rgb image
        self._traj_action = []          # action
        self._traj_reward = []          # reward
        self._traj_next_obs = {}        # observation
        self._traj_next_rgb = []        # rgb image
        
    def update_obs(self, prev: bool, obs):
        if prev:
            target = self._traj_prev_obs
        else:
            target = self._traj_next_obs
        
        if isinstance(obs, dict):
            for k, v in obs.items():
                nk = f'obs_{k}'
                if nk not in target:
                    target[nk] = []
                target[nk].append(v)
        else:
            k = 'obs'
            if k not in target:
                target[k] = []
            target[k].append(obs)
            
        
    def add_step(self, prev_obs, prev_rgb, action, reward, next_obs, next_rgb, done):
        
        self._acc_reward += reward
        
        '''
        Add to trajectory
        '''
        if prev_rgb.ndim == 3:
            prev_rgb = prev_rgb[None]
        if next_rgb.ndim == 3:
            next_rgb = next_rgb[None]
            
        self.update_obs(True, prev_obs)
        self._traj_prev_rgb.append(prev_rgb)
        self._traj_action.append(action)
        self._traj_reward.append(reward)
        self.update_obs(False, next_obs)
        self._traj_next_rgb.append(next_rgb)
        
        if done:
            if self._acc_reward > self._max_acc_reward:
                self._max_acc_reward = self._acc_reward
                
                traj_dir = os.path.join(self.logdir, f'{self._traj_cnt:06d}')
                os.makedirs(traj_dir, exist_ok=True)
                
                '''
                save observation
                '''
                for k, v in self._traj_prev_obs.items():
                    save_v = np.concatenate(v[:-1], axis=0)
                    np.save(os.path.join(traj_dir, f'prev_{k}.npy'), np.array(save_v))
                for k, v in self._traj_next_obs.items():
                    save_v = np.concatenate(v[:-1], axis=0)
                    np.save(os.path.join(traj_dir, f'next_{k}.npy'), np.array(save_v))                    
                
                '''
                save rgb images, actions, rewards
                '''
                # exclude last element, because it could be the terminal state
                prev_rgb = np.concatenate(self._traj_prev_rgb[:-1], axis=0)
                action = np.concatenate(self._traj_action[:-1], axis=0)
                reward = np.concatenate(self._traj_reward[:-1], axis=0)
                next_rgb = np.concatenate(self._traj_next_rgb[:-1], axis=0)
                
                np.save(os.path.join(traj_dir, 'prev_rgb.npy'), prev_rgb)
                np.save(os.path.join(traj_dir, 'action.npy'), action)
                np.save(os.path.join(traj_dir, 'reward.npy'), reward)
                np.save(os.path.join(traj_dir, 'next_rgb.npy'), next_rgb)
                
                # zip the files
                shutil.make_archive(traj_dir, 'zip', traj_dir)
                print(f"===== Saved trajectory at {traj_dir}.zip")
                
                # remove the directory
                shutil.rmtree(traj_dir)
                
                self._traj_cnt += 1
                
                # Reset trajectory
                self._traj_prev_obs = {}
                self._traj_prev_rgb = []
                self._traj_action = []
                self._traj_reward = []
                self._traj_next_obs = {}
                self._traj_next_rgb = []
                self._acc_reward = 0.0