#from mujoco_py import GlfwContext
#GlfwContext(offscreen=True)
import warnings,pdb

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import utils
import torch
from dm_env import specs

import metaworld_env as mw

from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import wandb
import math
import re,tqdm

from PIL import Image
from einops import rearrange

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = (9, 84, 84) #obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        if self.cfg.use_wandb:
            exp_name = '_'.join([cfg.reward_model, cfg.task_name, str(cfg.seed), 'ft_' + str(cfg.online_finetune_every_step), 'of_' + str(cfg.online_finetune)])
            group_name = re.search(r'\.(.+)\.', cfg.agent._target_).group(1)
            wandb.init(project="PROGRESSOR",
                       group=group_name,
                       name=exp_name,
                       config=cfg)
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self._discount = cfg.discount
        self._discount_alpha = cfg.discount_alpha
        self._discount_alpha_temp = cfg.discount_alpha_temp
        self._discount_beta = cfg.discount_beta
        self._discount_beta_temp = cfg.discount_beta_temp
        self._nstep = cfg.nstep
        self._nstep_alpha = cfg.nstep_alpha
        self._nstep_alpha_temp = cfg.nstep_alpha_temp
        self.setup()
        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=self.cfg.use_wandb)
        # create envs
        self.train_env = mw.make(self.cfg, self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed, self.cfg.reward_model, self.cfg.online_finetune)
        self.eval_env = mw.make(self.cfg, self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed, self.cfg.reward_model, False)
        self.eval_env.v2r_reward_model = self.train_env.v2r_reward_model
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1, ), np.float32, 'reward'),
                      specs.Array((1, ), np.float32, 'discount'),
                      specs.Array((1,), np.float32, 'env_reward')
                      )

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')
        self.replay_loader, self.buffer = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers, self.cfg.save_snapshot,
            math.floor(self._nstep + self._nstep_alpha),
            self._discount - self._discount_alpha - self._discount_beta)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None,
            render_reward=self.cfg.render_reward_on_video)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None,
            render_reward=self.cfg.render_reward_on_video)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def discount(self):
        return self._discount - self._discount_alpha * math.exp(
            -self.global_step /
            self._discount_alpha_temp) - self._discount_beta * math.exp(
                -self.global_step / self._discount_beta_temp)

    @property
    def nstep(self):
        return math.floor(self._nstep + self._nstep_alpha *
                          math.exp(-self.global_step / self._nstep_alpha_temp))

    def update_buffer(self):
        #self.buffer.update_discount(self.discount)
        self.buffer.update_nstep(self.nstep)
        return

    def eval(self):
        step, episode, total_reward, total_sr = 0, 0, 0, 0
        ### Detla
        env_total_reward = 0
        ###
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        self.video_recorder.init(self.eval_env, enabled=True)
        while eval_until_episode(episode):
            total_reward_this_episode = 0
            env_total_reward_this_episode = 0
            episode_sr = False
            time_step = self.eval_env.reset()
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                episode_sr = episode_sr or time_step.success
                cur_reward = time_step.reward
                total_reward_this_episode += cur_reward

                ### Detla
                cur_env_reward = time_step.env_reward
                env_total_reward_this_episode += cur_env_reward
                # Render the current reward/env_reward at the frame and the total reward/env_reward up to the frame
                self.video_recorder.record(self.eval_env, cur_reward, total_reward_this_episode, cur_env_reward, env_total_reward_this_episode)
                ###
                step += 1

            total_sr += episode_sr
            episode += 1

            # if self.global_frame > 1000000 and success > 0 and save_num < 4:
            #     imageio.mimsave(video_path, frames, format='GIF', duration = 40)
            #     save_num += 1
            #     print(video_path)
        self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_success_rate', total_sr / episode)
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

            ### Detla
            log('episode_env_reward', env_total_reward / episode)
            ###

    def train(self):
        # predicates
        reply_data = None
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward, episode_sr = 0, 0, False
        ### Detla
        print('Training Agent...')
        episode_env_reward = 0
        ###

        time_step = self.train_env.reset()
        ### Detla
        self.train_video_recorder.init(time_step.observation) 
        ####

        self.replay_storage.add(time_step)
        metrics = None

        self.train_env.eval_reward_model(f'eval_pretrain_final.png')

        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # reset env
                time_step = self.train_env.reset()

                self.replay_storage.add(time_step)                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_success_rate', episode_sr)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                        ### Detla
                        log('episode_env_reward', episode_env_reward)
                        ###

                # reset env
                time_step = self.train_env.reset()
                ### Detla
                self.train_video_recorder.init(time_step.observation) 
                ###

                self.replay_storage.add(time_step)
                # try to save snapshot
                if self.cfg.save_snapshot and self.global_step % 1e4 == 0:
                    self.save_snapshot('snapshot.pt', {'model_dict':self.train_env.v2r_reward_model.state_dict(),
                                                       'optim_model_dict':self.train_env.v2r_reward_optimizer.state_dict(),
                                                       })
                    self.save_snapshot('snapshot1.pt', {'model_dict':self.train_env.v2r_reward_model.state_dict(),
                                                       'optim_model_dict':self.train_env.v2r_reward_optimizer.state_dict()
                                                       })
                episode_sr = False
                episode_step = 0
                episode_reward = 0
                ### Detla
                episode_env_reward = 0
                ###

            # try to evaluate
            if eval_every_step(self.global_step) and self.global_step != 0:
            #if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):

                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step) and self.global_step % self.cfg.update_every_steps == 0:
                reply_data = next(self.replay_iter)


                if self.global_step % self.cfg.online_finetune_every_step == 0 and self.cfg.online_finetune:
                    self.train_env.online_update(reply_data)

                reply_data[2] = self.train_env.recalculate_reward_from_buffer(reply_data)
                metrics = self.agent.update(
                    reply_data, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
            # take env step
            time_step = self.train_env.step(action)
            cur_reward = time_step.reward
            episode_reward += cur_reward

            ### Detla
            cur_env_reward = time_step.env_reward
            episode_env_reward += cur_env_reward
            # Render the current reward/env_reward at the frame and the total reward/env_reward up to the frame
            self.train_video_recorder.record(time_step.observation, cur_reward, episode_reward, cur_env_reward, episode_env_reward)  
            ###
            episode_sr = episode_sr or time_step.success
            self.replay_storage.add(time_step, init_frame = self.train_env.init_frame, goal_frame = self.train_env.goal)

            episode_step += 1
            self._global_step += 1

    def save_snapshot(self, name = 'snapshot.pt', add_dict = {}):
        snapshot = self.work_dir / name
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(add_dict)
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, snapshot):
        #snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            if 'dict' not in k:
                self.__dict__[k] = v
        self.train_env.v2r_reward_model.load_state_dict(payload['model_dict'])
        self.train_env.v2r_reward_optimizer.load_state_dict(payload['optim_model_dict'])

@hydra.main(config_path='cfgs', config_name='config')
def main(cfgs):
    from train_mw import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfgs)
    resume_from_other_task = True
    try:
        snapshot = root_dir / 'snapshot.pt'
        if snapshot.exists():
            print(f'resuming: {snapshot}')
            workspace.load_snapshot(snapshot)
            print(f'finish resuming: {snapshot}')
            if resume_from_other_task: workspace.__dict__['_global_episode'] = 0; workspace.__dict__['_global_step'] =0
    except:
        snapshot = root_dir / 'snapshot.pt1'
        if snapshot.exists():
            print(f'resuming: {snapshot}')
            workspace.load_snapshot(snapshot)
            print(f'finish resuming: {snapshot}')
    workspace.train()


if __name__ == '__main__':
    main()
