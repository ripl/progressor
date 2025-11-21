import os
import gym
import numpy as np
from dm_env import StepType, specs
import dm_env
import numpy as np
from gym import spaces
from typing import Any, NamedTuple
from collections import deque

#### ADDED
import sys
sys.path.append('../Video2Reward/')
import simple_reward_model as v2r_model
import model_vip as vip_model
import torch
nn = torch.nn
import os,copy
from torchvision import transforms
import torchvision
import glob,tqdm
from metaworld_experts import get_expert_policy
from PIL import Image
from torch import distributions
import matplotlib.pyplot as plt

from dataloader_for_online_finetune import Frameloader, SegmentFrameloader

####
class MetaWorld:
    def __init__(
        self,
        name,
        seed=None,
        action_repeat=1,
        size=(64, 64),
        camera=None,
        render_step_img = True,
    ):
        import metaworld
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
        )

        os.environ["MUJOCO_GL"] = "egl"

        task = f"{name}-v2-goal-observable"
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed)
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat
        self._render_step_img = render_step_img
        self._camera = camera

        ### ADDED
        self.task_name = name
        ###

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(action["action"])
            success += float(info["success"])
            reward += rew or 0.0
        success = min(success, 1.0)
        assert success in [0.0, 1.0]
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ) if self._render_step_img else None,
            "state": state,
            "success": success,
        }
        return obs

    def reset(self):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        state = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ),
            "state": state,
            "success": False,
        }
        return obs

class NormalizeAction:
    def __init__(self, env, key="action"):
        self._env = env
        self._key = key
        space = env.act_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})

class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs["is_last"] = True
            self._step = None
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    success: Any

    ### ADDED
    env_reward: Any
    time_step: Any
    ###

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)

class metaworld_wrapper():
    def __init__(self, cfg, env, nstack, reward_model, online_finetune):
        self.cfg = cfg
        self._env = env
        self.nstack = 3
        cur_task =  self._env._env._env.task_name
        self.expert_policy = get_expert_policy(cur_task)
        wos = env.obs_space['image']  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros(low.shape, low.dtype)

        self.observation_space = spaces.Box(low=np.transpose(low, (2, 0, 1)), high=np.transpose(high, (2, 0, 1)), dtype=np.uint8)

        #### ADDED
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_frame = None 

        self.mean=np.array([0.485, 0.456, 0.406])
        self.std=np.array([0.229, 0.224, 0.225])
        self.v2r_transform = transforms.Compose([
            transforms.Resize(84),                                        
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.online_geo_transform = torchvision.transforms.RandomResizedCrop(size = 84, scale=(0.8, 1.0))

        data_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std)
            ])
        self.to_tensor = transforms.ToTensor()
        self.online_finetune = online_finetune

        if reward_model == 'ours':
            self.v2r_reward_model = v2r_model.Model(model_type="resnet34", cfg = cfg)   #Modified
            self.v2r_reward_model.to(self.device)
            self.v2r_reward_optimizer = torch.optim.AdamW(self.v2r_reward_model.parameters(), lr = 5e-5)
            
            def add_sn(m):
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    return torch.nn.utils.spectral_norm(m)
                else:
                    return m
                
            reward_path = '/' + os.path.join(*os.getcwd().split('/')[:-2]) + f'/pretrained_reward/all_tasks.pth' #TODO config

            if os.path.exists(reward_path):
                print("Loading our reward model weight...")
                ckpt = torch.load(reward_path)
                self.v2r_reward_model.load_state_dict(ckpt['state_dict'])
                self.v2r_reward_optimizer.load_state_dict(ckpt['optim_state_dict'])
            else:
                pretrain_dataset = Frameloader(self.cfg.exp_data_path,
                        transforms = data_transforms, 
                        geo_transforms=self.online_geo_transform,
                        multi_task=True,
                        randomized=True
                        )
                self.pretrain_iter = iter(torch.utils.data.DataLoader(pretrain_dataset,
                                                        batch_size=256,
                                                        num_workers=6,
                                                        pin_memory=True,
                                                        ))
                
                self.eval_dataset = SegmentFrameloader(os.path.join(self.cfg.exp_data_path, cur_task, 'test'),  #TODO all tasks
                                        train_test_split_ratio = 1.0,
                                        transforms=data_transforms,
                                        normalize_trajectory=True,
                                        segment_size = 101,
                                        frame_internal = 1)

                self.pretrain_reward(reward_path)    


        if self.online_finetune:
            self.v2r_reward_model.train()

            dataset = Frameloader(self.cfg.exp_data_path,
                        transforms = data_transforms, 
                        geo_transforms=self.online_geo_transform,
                        multi_task=False,
                        randomized=False,
                        cur_task=cur_task
                        )
            self.online_iter = iter(torch.utils.data.DataLoader(dataset,
                                                 batch_size=256,
                                                 num_workers=6,
                                                 pin_memory=True,
                                                 ))
        self.eval_dataset = SegmentFrameloader(os.path.join(self.cfg.exp_data_path, cur_task, 'test'),  #TODO all tasks
                            train_test_split_ratio = 1.0,
                            transforms=data_transforms,
                            normalize_trajectory=True,
                            segment_size = 101,
                            frame_internal =1)
        
        self.goal = None
        ####

    def pretrain_reward(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        for i in tqdm.tqdm(range(10000)):
            self.online_update(None)
            if i % 1000 == 0 and i != 0:
                self.eval_reward_model(f'eval_pretrain_{i:05d}.png')
        torch.save({'state_dict': self.v2r_reward_model.state_dict(),
                    'optim_state_dict': self.v2r_reward_optimizer.state_dict()}, save_path)

    @torch.no_grad()
    def eval_reward_model(self, save_path = 'eval.png'):
        self.v2r_reward_model.eval()
        minibatch = 16
        device_id = 'cuda'
        out_list = []
        for idx, (init_img, goal_img, mid_img, relative_position, delta_goal_init) in enumerate(tqdm.tqdm(self.eval_dataset)):
            seq_list = []
            init_img = init_img[None,...].to(device_id)
            goal_img = goal_img[None,...].to(device_id)
            mid_img = mid_img.to(device_id)
            feat_list = []
            for idx in range((mid_img.shape[0] + minibatch - 1) // minibatch):
                mid_img_batch = mid_img[idx * minibatch: (idx + 1) * minibatch]
                out = self.v2r_reward_model.compute_reward(torch.cat([init_img.expand(mid_img_batch.shape[0],-1,-1,-1), mid_img_batch,
                      goal_img.expand(mid_img_batch.shape[0],-1,-1,-1)], dim = 1))
                seq_list.append(out)

            seq = torch.cat(seq_list)
            seq = torch.nn.functional.interpolate(seq.reshape(1,1,-1), size = (100), mode = 'linear').reshape(-1)

            out_list.append(seq)
        seq = torch.stack(out_list)
        seq_mean = seq.mean(dim = 0).cpu().data.numpy()
        fig, ax = plt.subplots()
        ax.plot(np.arange(100), seq_mean, linewidth = 4.0)
        ax.fill_between(np.arange(100), seq_mean - seq.std(dim = 0).cpu().data.numpy(), seq_mean + seq.std(dim = 0).cpu().data.numpy(), color='#888888', alpha=0.4)
        plt.savefig(save_path)
        plt.close()
        self.v2r_reward_model.train()

    def observation_spec(self):
        return specs.BoundedArray(self.observation_space.shape,
                                  np.uint8,
                                  0,
                                  255,
                                  name='observation')

    def action_spec(self):
        return specs.BoundedArray(self._env.act_space['action'].shape,
                                  np.float32,
                                  self._env.act_space['action'].low,
                                  self._env.act_space['action'].high,
                                  'action')

    def reset(self):
        time_step = self._env.reset()
        obs = time_step['image']
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        ### ADDED
        self.init_frame = None
        self.goal = None
        
        time_step['image'] = None
        return ExtendedTimeStep(observation=np.transpose(self.stackedobs, (2, 0, 1)),
                                step_type=StepType.FIRST,
                                action=np.zeros(self.action_spec().shape, dtype=self.action_spec().dtype),
                                reward=0.0,
                                discount=1.0,
                                success = time_step['success'],
                                env_reward=0.0,     ### ADDED
                                time_step=time_step
                                )

    def render_goal_image(self, time_step, max_step = 1000):
        time_step = copy.deepcopy(time_step)
        env_for_goal_img = copy.deepcopy(self._env)
        inner_env_for_goal_img = env_for_goal_img._env._env
        inner_env_for_goal_img._render_step_img = False
        while not time_step['is_last']:
            for i in range(max_step):
                action = self.expert_policy.get_action(time_step['state'])
                time_step = env_for_goal_img.step({'action':action})
                if time_step['is_last']:
                    break
        goal_img = inner_env_for_goal_img._env.sim.render(
            *inner_env_for_goal_img._size, mode="offscreen", camera_name=inner_env_for_goal_img._camera)
        return self.v2r_transform(torch.from_numpy(goal_img).permute(2,0,1).float() * 1.0 / 255)

    @torch.no_grad()
    def recalculate_reward_from_buffer(self, non_exp_data_reply):
        self.v2r_reward_model.eval()
        no_exp_data = ((non_exp_data_reply[0][:,:3] * 1.0 / 255) - self.mean.reshape(1,3,1,1)) / self.std.reshape(1,3,1,1)
        no_exp_init_frame = non_exp_data_reply[-2]
        no_exp_goal_frame = non_exp_data_reply[-1]
        no_exp_data = torch.cat([no_exp_init_frame, no_exp_data, no_exp_goal_frame], dim = 1)
        reward = self.v2r_reward_model.compute_reward(no_exp_data.cuda().float())
        return reward

    def online_update(self, non_exp_data_reply = None, mu_push_back_factor = 0.9, var_push_back_factor = 1.0):
        self.v2r_reward_model.train()
        
        self.v2r_reward_optimizer.zero_grad()
        exp_data, exp_gt_label, exp_gt_var = next(self.online_iter) if non_exp_data_reply is not None else next(self.pretrain_iter)
        #visualize_stack_img(exp_data)
        exp_pred = self.v2r_reward_model(exp_data.cuda().float())[0]
        lbl = distributions.Normal(exp_gt_label.cuda().reshape(-1), exp_gt_var.cuda().reshape(-1))
        loss = distributions.kl.kl_divergence(lbl, exp_pred).mean()
        loss.backward()
        loss1 = loss.item()
        if non_exp_data_reply is not None:
            no_exp_data = ((non_exp_data_reply[0][:,:3] * 1.0 / 255) - self.mean.reshape(1,3,1,1)) / self.std.reshape(1,3,1,1)
            no_exp_init_frame = non_exp_data_reply[-2]
            no_exp_goal_frame = non_exp_data_reply[-1]
            no_exp_data = torch.cat([no_exp_init_frame, no_exp_data, no_exp_goal_frame], dim = 1)
            no_exp_data = self.online_geo_transform(no_exp_data)
            no_exp_pred = self.v2r_reward_model(no_exp_data.cuda().float())[0]
            lbl = distributions.Normal(no_exp_pred.loc.detach().data * mu_push_back_factor, no_exp_pred.scale.detach().data * var_push_back_factor)
            loss = 0.01 * distributions.kl.kl_divergence(lbl, no_exp_pred).mean()
            loss.backward()
            loss2 = 10 * loss.item()
            print(loss1, loss2)
        else:
            print(loss1)
        torch.nn.utils.clip_grad_norm_(self.v2r_reward_model.parameters(), 1)
        
        self.v2r_reward_optimizer.step()
        
        self.v2r_reward_model.eval()

    def step(self, action):
        action = {'action':action}
        time_step = self._env.step(action)
        obs = time_step['image']
        self.stackedobs = np.roll(self.stackedobs, shift=-obs.shape[-1], axis=-1) #
        self.stackedobs[..., -obs.shape[-1]:] = obs

        ### ADDED
        if self.init_frame is None:
            self.init_frame = self.v2r_transform(self.to_tensor(obs / 255.0))
            self.goal = self.render_goal_image(time_step)

        ####
        transform_obs = self.v2r_transform(self.to_tensor(obs / 255.0))
        triplate = torch.cat([self.init_frame, transform_obs ,self.goal]).unsqueeze(0).to(self.device).float()
        pred_reward = self.v2r_reward_model.compute_reward(triplate).item()

        if time_step['is_first']:
            step_type = StepType.FIRST
        elif time_step['is_last']:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        return ExtendedTimeStep(observation=np.transpose(self.stackedobs, (2, 0, 1)),
                                step_type=step_type,
                                action=action['action'],
                                reward=pred_reward,                     # PROGRESSOR estimated Reward
                                discount=1.0,
                                success = time_step['success'],
                                env_reward=time_step['reward'],         # Environment Reward
                                time_step = time_step,
                                )

    def get_pixels_with_width_height(self, width, height):
        camera = self._env._env._env._camera
        img = self._env._env._env._env.sim.render(width, height, mode='offscreen',camera_name=camera)
        return img

def make(cfg, name, frame_stack, action_repeat, seed, reward_model = 'ours', online_finetune = False):
    env = MetaWorld(name, seed,action_repeat, (84, 84), 'corner2')
    env = NormalizeAction(env)
    env = TimeLimit(env, 100)
    env = metaworld_wrapper(cfg, env, frame_stack, reward_model, online_finetune)
    return env
