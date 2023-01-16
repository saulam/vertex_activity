import numpy as np
import pickle as pk
import random
import torch
from torch.utils.data import Dataset
from glob import glob

# dataset
class VAdataset(Dataset):
    def __init__(self, root, shuffle=False, **kwargs):
        '''Initialiser for VAdataset class'''

        self.root = root
        self.data_files = self.processed_file_names
        if shuffle:
            random.shuffle(self.data_files)
        self.total_events = len(self.data_files)

        with open("/".join(root.split("/")[:-1]) + "/data.p", "rb") as fd:
            charges, ini_pos, ini_P, Es, total_charges = pk.load(fd)

        # all parameters
        self.min_charge = charges.min()
        self.max_charge = charges.max()
        self.min_tCharge = total_charges.min()
        self.max_tCharge = total_charges.max()
        self.min_pos = ini_pos.min()
        self.max_pos = ini_pos.max()
        self.min_P = ini_P.min()
        self.max_P = ini_P.max()
        self.min_E = Es.min()
        self.max_E = Es.max()
        self.range_pos = self.max_pos - self.min_pos
        self.range_P = self.max_P - self.min_P

        self.source_range = (-1, 1)
        self.target_range = (0, 1)

        self.m_proton = 938.27208816  # mass of the proton in MeV

        self.min_charge_new = 0
        self.max_charge_new = charges.max()

    @property
    def processed_dir(self):
        return f'{self.root}'

    @property
    def processed_file_names(self):
        return sorted(glob(f'{self.processed_dir}/*.npz'))

    def __len__(self):
        return self.total_events

    def collate_fn(self, batch):
        img_batch = np.array([event['full_image'] for event in batch if event['full_image'] is not None])
        params_batch = np.array([np.concatenate([event['initPosition'], event['initMomentum']]) \
                                 for event in batch if event['initPosition'] is not None])
        signal_cubes = np.array([event['signal_cubes'] for event in batch if event['signal_cubes'] is not None])

        img_batch = torch.tensor(img_batch).float()
        params_batch = torch.tensor(params_batch).float()
        signal_cubes = torch.tensor(signal_cubes).long()

        return img_batch, params_batch, signal_cubes

    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])

        # retrieve input
        sparse_image = data['sparse_image']  # array of shape (Nx2) [points vs (1d pos, charge)]
        initPosition = data['initPosition']  # array with initial position (x1, y1, z1)
        finalPosition = data['finalPosition']  # array with initial position (xN, yN, zN)
        initMomentum = data['initMomentum']  # array momentum vector (Px, Py, Pz, E)

        if sparse_image.shape[0] == 0:
            del data
            return {'signal_cubes': None,
                    'full_image': None,
                    'initPosition': None,
                    'finalPosition': None,
                    'initMomentum': None}

        # reconstruct the image from sparse points to a 7x7x7 volume
        full_image = np.zeros(shape=(7 * 7 * 7))
        full_image[sparse_image[:, 0].astype(int)] = sparse_image[:, 1]

        # 5x5x5 volume
        full_image = full_image.reshape(7, 7, 7)
        full_image = full_image[1:-1, 1:-1, 1:-1]
        full_image = full_image.reshape(-1)
        totalCharge = full_image.sum()

        full_image = full_image.reshape(-1)

        # normalise
        signal_cubes = full_image > 0
        full_image = np.interp(full_image.ravel(), (self.min_charge_new, self.max_charge_new), self.target_range).reshape(
            full_image.shape)
        initPosition = np.interp(initPosition.ravel(), (self.min_pos, self.max_pos), self.source_range).reshape(initPosition.shape)
        initMomentum[:3] = np.interp(initMomentum[:3].ravel(), (self.min_P, self.max_P), self.source_range).reshape(
            initMomentum[:3].shape)
        initMomentum[3] = np.interp(initMomentum[3].ravel(), (self.min_E, self.max_E), self.source_range).reshape(
            initMomentum[3].shape)

        del data
        return {'signal_cubes': signal_cubes,
                'full_image': full_image,
                'initPosition': initPosition,
                'finalPosition': finalPosition,
                'initMomentum': initMomentum}