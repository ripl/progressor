import os, glob
import numpy as np
import torch,pdb
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import IterableDataset
from PIL import Image
import random

class Frameloader(IterableDataset):
    ''' Dataset class for loading expert trajectories.
    '''
    def __init__(self,
                 root_dir,
                 test=False,
                 normalize_trajectory=True,
                 transforms=None,
                 geo_transforms = None,
                 subset_len=1,
                 randomized=True,
                 min_frames_between=1,
                 multi_task= True,
                 cur_task=''
                 ):
        if not multi_task: assert cur_task !=''

        dataset = 'test' if test else 'train'
        self.path = os.path.join(root_dir, f'*/{dataset}/*') if multi_task else os.path.join(root_dir, cur_task, f'{dataset}/*')
        self.ds = sorted(glob.glob(self.path))
        self.normalize_trajectory = normalize_trajectory
        self.transforms = transforms
        self.geo_transforms = geo_transforms
        self.subset_len = subset_len
        self.expert_traj_lens = {i: len(glob.glob(self.ds[i] + '/*.png')) for i in range(len(self.ds))}
        self.randomized = randomized
        self.min_frames_between = min_frames_between

    def __iter__(self):
        while True:
            yield self._sample()

    def _sample(self):
        index = np.random.randint(len(self.ds))
        # Randomly select start and end frames of the trajectory
        if self.randomized:
            #init_frame = np.random.randint(0, self.expert_traj_lens[index] - self.min_frames_between, self.subset_len)
            #goal_frame = np.random.randint(init_frame + self.min_frames_between, self.expert_traj_lens[index], self.subset_len)
            init_frame = np.random.randint(0, self.expert_traj_lens[index] - self.min_frames_between, self.subset_len).item()
            goal_frame = np.random.randint(init_frame + self.min_frames_between, self.expert_traj_lens[index], self.subset_len).item()
            mid_frame = np.random.randint(init_frame, goal_frame + 1)
        # Fix start and end frames of the trajectory as the first and last frames the dataset
        else:
            init_frame = 0
            goal_frame = self.expert_traj_lens[index] - 1
            mid_frame = np.random.randint(init_frame, goal_frame + 1)

        # Length of the trajectory
        delta_goal_init =  goal_frame - init_frame
        # Relative position of the mid frame in the trajectory from 0 - 1
        relative_position = (mid_frame - init_frame) / delta_goal_init
        # Get batch of frames, applying transforms if given

        cur = torch.vstack(
            (self.transforms(Image.open(os.path.join(self.ds[index], str(init_frame) + '.png'))),
             self.transforms(Image.open(os.path.join(self.ds[index], str(mid_frame) + '.png'))),
             self.transforms(Image.open(os.path.join(self.ds[index], str(goal_frame) + '.png')))))
        cur = self.geo_transforms(cur)
        # Return the frames and their relative position in the trajectory
        if self.normalize_trajectory:
            return cur, relative_position, np.round(np.sqrt(1/delta_goal_init), 3)
        # Return the frames and their absolute position in the trajectory
        return cur, np.round(relative_position * delta_goal_init), np.ones(self.subset_len)

class SegmentFrameloader(Dataset):
    ''' Dataset class for loading expert trajectories.
    '''
    def __init__(self,
                 root_dir,
                 train_test_split_ratio = 1.0,
                 normalize_trajectory=True,
                 transforms=None,
                 frame_internal = 60,
                 segment_size = 5000,
                 max_seg_per_folder = 5
                 ):
        self.ds = [folder for folder in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(folder)]
        self.ds.sort()
        self.normalize_trajectory = normalize_trajectory
        self.transforms = transforms
        self.frame_internal = frame_internal
        #self.frame_name = {}
        self.frame_name = {i: sorted(glob.glob(self.ds[i] + '/*.jpg')) for i in range(len(self.ds))}
        if len(self.frame_name[0]) == 0:
            self.frame_name = {i: [self.ds[i] + f'/{idx}.png' for idx in range(len(glob.glob(self.ds[i] + '/*')))] for i in range(len(self.ds))}
        self.expert_traj_lens = {i: len(self.frame_name[i]) for i in range(len(self.ds))}
        self.segment_info = []
        for key, val in self.expert_traj_lens.items():
            segment_info = []
            for i in np.arange((val + segment_size - 1) // segment_size):
                start_index = i * segment_size
                end_index = min((i + 1) * segment_size, val)
                segment_info.append([key, start_index, end_index])
            random.shuffle(segment_info)
            self.segment_info += segment_info[:max_seg_per_folder]

    def __len__(self):
        return len(self.segment_info)

    def __getitem__(self, index):
        index = index
        segment_index, init_frame, goal_frame = self.segment_info[index]
        frame_name_list = self.frame_name[segment_index]
        mid_frame = np.arange(init_frame, goal_frame)[::self.frame_internal]
        # Length of the trajectory
        delta_goal_init =  goal_frame - init_frame
        # Relative position of the mid frame in the trajectory from 0 - 1
        relative_position = (mid_frame - init_frame) / delta_goal_init

        # Get batch of frames, applying transforms if given
        init_img = self.transforms(Image.open(frame_name_list[init_frame]))
        goal_img = self.transforms(Image.open(frame_name_list[goal_frame - 1]))
        mid_img = torch.stack([self.transforms(Image.open(frame_name_list[idx])) for idx in mid_frame])
        return init_img, goal_img, mid_img, relative_position, delta_goal_init
