import os, glob,tqdm
import numpy as np
import torch,pdb,time
from torch.utils.data import Dataset
from PIL import Image
import random

class Frameloader(Dataset):
    ''' Dataset class for loading expert trajectories.
    '''
    def __init__(self,
                 root_dir,
                 train = True,
                 train_test_split_ratio = 0.8,
                 normalize_trajectory=True,
                 transforms=None,
                 geo_augment = None,
                 randomized=False,
                 min_frames_between=1,
                 max_frame_gap = 2000,
                 frame_sample_mode = 'triplet',
                 expand_factor = 10
                 ):
        self.root_dir = root_dir
        self.expand_factor = expand_factor
        assert frame_sample_mode in ['triplet', 'contrastive', 'vip']
        self.ds = [folder for folder in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(folder)]
        self.ds.sort()
        if not os.path.exists('epic_kitchen_data_list.pth'):
            self.img_name = {}
            for ds in tqdm.tqdm(self.ds):
                img_list = sorted(glob.glob(ds + '/*.jpg'))
                self.img_name[ds.split('/')[-1]] = [img_path.split('/')[-1] for img_path in img_list]
            torch.save(self.img_name, 'epic_kitchen_data_list.pth')
        else:
            self.img_name = torch.load('epic_kitchen_data_list.pth')
            assert len(self.img_name.keys()) == len(self.ds)
        train_test_split_idx = int(train_test_split_ratio * len(self.ds))
        if train:
            self.ds = self.ds[:train_test_split_idx]
        else:
            self.ds = self.ds[train_test_split_idx:]
        self.normalize_trajectory = normalize_trajectory
        self.transforms = transforms
        self.geo_transform = geo_augment
        self.randomized = randomized
        self.min_frames_between = min_frames_between
        self.max_frame_gap = max_frame_gap
        self.frame_sample_mode = frame_sample_mode
        self.folder_clip_cumsum = self.compute_folder_clip_num()

    def compute_folder_clip_num(self):
        folder_clip = []
        for ds in self.ds:
            folder_clip.append((len(self.img_name[ds.split('/')[-1]]) + 2000 - 1)// 2000)
        folder_clip_cumsum = np.cumsum(np.array(folder_clip))
        return folder_clip_cumsum.astype(np.int32)

    def __len__(self):
        return self.folder_clip_cumsum[-1] * self.expand_factor

    def __getitem__(self, index):
        index = index % self.folder_clip_cumsum[-1]
        index = np.searchsorted(self.folder_clip_cumsum, index, side = 'right')
        folder_name = self.ds[index].split('/')[-1]
        frame_num = len(self.img_name[folder_name])
        # Randomly select start and end frames of the trajectory
        init_frame = np.random.randint(0, frame_num  - self.min_frames_between)
        goal_frame = np.random.randint(init_frame + self.min_frames_between, min(frame_num, init_frame + self.max_frame_gap))

        # Length of the trajectory
        delta_goal_init =  goal_frame - init_frame
        # Relative position of the mid frame in the trajectory from 0 - 1

        # Get batch of frames, applying transforms if given
        all_frames = [self.transforms(Image.open(self.ds[index]  + '/' + self.img_name[folder_name][init_frame]))]
        if self.frame_sample_mode == 'triplet':
            mid_frame = np.random.randint(init_frame, goal_frame + 1)
            all_frames.append(self.transforms(Image.open(self.ds[index]  + '/' + self.img_name[folder_name][mid_frame])))
        elif self.frame_sample_mode == 'vip':
            mid_frame = np.random.randint(init_frame, goal_frame + 1)
            all_frames.append(self.transforms(Image.open(self.ds[index]  + '/' + self.img_name[folder_name][mid_frame])))
            mid_frame = min(mid_frame + 1, goal_frame)
            all_frames.append(self.transforms(Image.open(self.ds[index]  + '/' + self.img_name[folder_name][mid_frame])))
        all_frames.append(self.transforms(Image.open(self.ds[index]  + '/' + self.img_name[folder_name][goal_frame])))
        all_frames = torch.stack(all_frames, axis=0).flatten(0,1)
        if self.geo_transform is not None:
            all_frames = self.geo_transform(all_frames)
            all_frames = all_frames.reshape(-1, 3, all_frames.shape[1], all_frames.shape[2])
        # Return the frames and their relative position in the trajectory
        #if self.normalize_trajectory:
        return all_frames, np.array([init_frame, mid_frame, goal_frame])
        # Return the frames and their absolute position in the trajectory
        #return all_frames, np.round(relative_position * delta_goal_init), np.ones(self.subset_len)

class R3MFrameloader(Frameloader):
    ''' Dataset class for loading expert trajectories.
    '''
    def __init__(self,
                 root_dir,
                 train = True,
                 train_test_split_ratio = 0.8,
                 normalize_trajectory=True,
                 transforms=None,
                 geo_augment = None,
                 randomized=False,
                 min_frames_between=1,
                 max_frame_gap = 2000,
                 frame_sample_mode = 'triplet',
                 expand_factor = 10
                 ):
        super().__init__(root_dir = root_dir, train = train,
                        train_test_split_ratio = train_test_split_ratio,
                        normalize_trajectory = normalize_trajectory,
                        transforms = transforms,
                        geo_augment = geo_augment,
                        randomized = randomized,
                        min_frames_between = min_frames_between,
                        max_frame_gap = max_frame_gap,
                        frame_sample_mode = frame_sample_mode,
                        expand_factor = expand_factor)

    def __len__(self):
        return self.folder_clip_cumsum[-1] * self.expand_factor

    def __getitem__(self, index):
        index = index % self.folder_clip_cumsum[-1]
        index = np.searchsorted(self.folder_clip_cumsum, index, side = 'right')
        folder_name = self.ds[index].split('/')[-1]
        vidlen = len(self.img_name[folder_name])
        # Randomly select start and end frames of the trajectory
        #init_frame = np.random.randint(0, frame_num  - self.min_frames_between)
        #goal_frame = np.random.randint(init_frame + self.min_frames_between, min(frame_num, init_frame + self.max_frame_gap))
        start_ind = np.random.randint(1, 2 + int(self.alpha * vidlen))
        end_ind = np.random.randint(int((1-self.alpha) * vidlen)-1, vidlen)
        s1_ind = np.random.randint(2, vidlen)
        s0_ind = np.random.randint(1, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen+1)
        all_frames = []
        for idx in [start_ind, end_ind, s0_ind, s1_ind, s2_ind]:
            all_frames.append(self.transforms(Image.open(self.ds[index]  + '/' + self.img_name[folder_name][idx])))
        all_frames = torch.stack(all_frames, axis=0)
        if self.geo_transform is not None:
            all_frames = all_frames.flatten(0,1)
            all_frames = self.geo_transform(all_frames)
            all_frames = all_frames.reshape(-1, 3, all_frames.shape[1], all_frames.shape[2])
        return all_frames

class SegmentFrameloader(Dataset):
    ''' Dataset class for loading expert trajectories.
    '''
    def __init__(self,
                 root_dir,
                 train,
                 train_test_split_ratio = 1.0,
                 normalize_trajectory=True,
                 transforms=None,
                 frame_internal = 60,
                 segment_size = 5000,
                 max_seg_per_folder = 5
                 ):
        self.ds = [folder for folder in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(folder)]
        self.ds.sort()
        train_test_split_idx = int(train_test_split_ratio * len(self.ds))
        if train:
            self.ds = self.ds[:train_test_split_idx]
        else:
            self.ds = self.ds[train_test_split_idx:]
        self.normalize_trajectory = normalize_trajectory
        self.transforms = transforms
        self.frame_internal = frame_internal
        #self.frame_name = {}
        self.frame_name = {i: sorted(glob.glob(self.ds[i] + '/*.jpg')) for i in range(len(self.ds))}
        if len(self.frame_name[0]) == 0:
            self.frame_name = {i: [self.ds[i] + f'/{idx}.png' for idx in range(100)] for i in range(len(self.ds))}
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

if __name__ == '__main__':
    # Initialize dataset
    dataset_train = SegmentFrameloader(
        root_dir='',
        train = False,
        train_test_split_ratio = 0.8,
        transforms=None,
        normalize_trajectory=True,
        segment_size = 5000,
        frame_internal = 60
        )
    dataset_train[0]
    dataset_train = Frameloader(
        root_dir='',
        train = True,
        train_test_split_ratio = 0.8,
        transforms=None,
        normalize_trajectory=True,
        subset_len=8,
        randomized=True)
