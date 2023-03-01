import numpy as np
import pickle as pk
import random
import torch
from torch.utils.data import Dataset
from glob import glob

# dataset
class VAdatasetFinal(Dataset):
    def __init__(self, root, source_range=(-1, 1), target_range=(0, 1), shuffle=False, **kwargs):
        '''Initialiser for VAdataset class'''

        self.root = root
        self.data_files = self.processed_file_names
        if shuffle:
            random.shuffle(self.data_files)
        self.total_events = len(self.data_files)

        with open("/scratch2/salonso/vertex_activity/data.p", "rb") as fd:
            charges, ini_pos, kes, thetas, phis = pk.load(fd)

        # all parameters
        self.min_charge = charges.min()
        self.max_charge = charges.max()
        self.min_pos = ini_pos.min()
        self.max_pos = ini_pos.max()
        self.min_KE = 0
        self.max_KE = 60
        self.min_theta = 0
        self.max_theta = np.pi
        self.min_phi = -np.pi
        self.max_phi = np.pi

        self.source_range = source_range
        self.target_range = target_range

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
        params_batch = np.array([np.concatenate([event['pos_ini'], event['ke'], event['theta'], event['phi']])
                                 for event in batch if event['pos_ini'] is not None])

        img_batch = torch.tensor(img_batch).float()
        params_batch = torch.tensor(params_batch).float()

        return img_batch, params_batch

    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])

        # retrieve input
        sparse_image = data['sparse_image']  # array of shape (Nx2) [points vs (1d pos, charge)]
        pos_ini = data['pos_ini']  # array with initial position (x1, y1, z1)
        ke = data['ke']  # Kinetic energy
        theta = data['theta']
        phi = data['phi']

        if sparse_image.shape[0] == 0:
            del data
            return {'full_image': None,
                    'pos_ini': None,
                    'ke': None,
                    'theta': None,
                    'phi': None}

        # reconstruct the image from sparse points to a 7x7x7 volume
        full_image = np.zeros(shape=(7 * 7 * 7))
        full_image[sparse_image[:, 0].astype(int)] = sparse_image[:, 1]

        # 5x5x5 volume
        full_image = full_image.reshape(7, 7, 7)
        full_image = full_image[1:-1, 1:-1, 1:-1]
        full_image = full_image.reshape(-1)


        # normalise
        full_image = np.interp(full_image.ravel(), (self.min_charge_new, self.max_charge_new), self.target_range).reshape(
            full_image.shape)
        pos_ini = np.interp(pos_ini.ravel(), (self.min_pos, self.max_pos), self.source_range).reshape(pos_ini.shape)
        ke = np.interp(ke, (self.min_KE, self.max_KE), self.source_range).reshape(1)
        theta = np.interp(theta, (self.min_theta, self.max_theta), self.source_range).reshape(1)
        phi = np.interp(phi, (self.min_phi, self.max_phi), self.source_range).reshape(1)

        del data
        return {'full_image': full_image,
                'pos_ini': pos_ini,
                'ke': ke,
                'theta': theta,
                'phi': phi}