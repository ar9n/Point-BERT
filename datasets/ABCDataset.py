import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class ABC(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        
        self.train_test_split = config.get('train_test_split', 0.8)
        
        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        self.rot = config.get('rot', False)

        self.num_chunks = config.get('chunks', 100)

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'ABC')
        
        self.file_list = []

        for root, dirs, files in os.walk(self.data_root):
            for f in files:
                if f.endswith('.npy'):
                    chunk = root.split('/')[-1]
                    chunk_idx = int(chunk[6:])
                    if chunk_idx < self.num_chunks:
                        self.file_list.append({
                            'chunk': chunk_idx,
                            'file_name': f,
                            'file_path': os.path.join(root, f)
                        })

        if not self.whole:
            if self.subset == 'train':
                self.file_list = self.file_list[:int(len(self.file_list) * self.train_test_split)]
            elif self.subset == 'test':
                self.file_list = self.file_list[int(len(self.file_list) * self.train_test_split):]  

        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'ABC')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    # This is the point where the point cloud with fixed size of 1024 is sampled randomly
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(sample['file_path']).astype(np.float32)
        sampled_ids = np.random.choice(data.shape[0], 1024, replace=True)
        data = data[sampled_ids]
        data = self.pc_norm(data)
        if self.rot:
            data = data @ rnd_rot()
        data = torch.from_numpy(data).float()
        return sample['chunk'], sample['file_name'], (data, 0)

    def __len__(self):
        return len(self.file_list)