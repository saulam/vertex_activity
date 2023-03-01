import numpy as np
import pickle as pk
import random
import torch
import re
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from glob import glob

# dataset
class VAdatasetDecoder(Dataset):
    def __init__(self, root, source_range=(-1, 1), target_range=(0, 1),
                 max_protons = 3, PAD_IDX=-2):
        '''Initialiser for VAdataset class'''

        self.root = root
        self.file_names = root
        self.PAD_IDX = PAD_IDX

        with open("/scratch2/salonso/vertex_activity/data_5M.p", "rb") as fd:
            charges, ini_pos, kes, thetas, phis, lookup_table, bin_edges = pk.load(fd)

        # all parameters
        self.cube_size = 10.27
        self.img_size = 7
        self.max_protons = max_protons
        self.min_charge = charges.min()
        self.max_charge = charges.max()
        self.min_pos = -self.cube_size-self.cube_size/2.
        self.max_pos = self.cube_size+self.cube_size/2.
        self.min_KE = 0
        self.max_KE = 60
        self.min_theta = 0
        self.max_theta = np.pi
        self.min_phi = -np.pi
        self.max_phi = np.pi
        self.lookup_table = lookup_table
        self.bin_edges = bin_edges

        self.source_range = source_range
        self.target_range = target_range

        self.m_proton = 938.27208816  # mass of the proton in MeV

        self.min_charge_new = 0
        self.max_charge_new = charges.max()

        self.indices = list(self.lookup_table.keys())
        self.total_events = len(self.indices)

    '''
    @property
    def processed_dir(self):
        return f'{self.root}'

    @property
    def processed_file_names(self):
        return sorted(glob(f'{self.processed_dir}/*.npz'))
    '''

    def __len__(self):
        return self.total_events

    def collate_fn(self, batch):
        img_batch, ini_pos, params_batch, is_next_batch = [], [], [], []

        for event in batch:
            if event['images'] is None:
                continue

            charge_sum = event['images'].sum(0)
            indexes = np.where(charge_sum)
            charges = charge_sum[indexes].reshape(-1, 1)
            indexes = np.stack(indexes, axis=1)
            image = torch.tensor(np.concatenate((indexes, charges), axis=1))
            pos_ini = torch.tensor(event['ini_pos'])
            params = torch.tensor(event['params'])
            is_next = torch.ones(size=(params.shape[0],))
            is_next[-1] = 0

            img_batch.append(image)
            ini_pos.append(pos_ini)
            params_batch.append(params)
            is_next_batch.append(is_next)

        assert len(img_batch) > 0

        img_batch = pad_sequence(img_batch, padding_value=self.PAD_IDX).float()
        ini_pos = torch.stack(ini_pos).float()
        params_batch = pad_sequence(params_batch, padding_value=self.PAD_IDX).float()
        is_next_batch = pad_sequence(is_next_batch, padding_value=self.PAD_IDX).long()

        return img_batch, ini_pos, params_batch, is_next_batch

    def __getitem__(self, idx):

        # get index for the lookup table
        index = self.indices[idx]
        candidates = self.lookup_table[index]

        # get from 1 to 5 protons with the same initial position
        events = []
        for event_id in random.sample(candidates, random.randint(1, min(self.max_protons, len(candidates)))):
            filepath = self.file_names.format(event_id)
            candidate = np.load(filepath)
            events.append(candidate)

        images, params, lens = [], [], []
        ext_x, ext_y, ext_z = np.random.randint(0, 2 + 1, 3)  # random shift (same for all the protons)
        # retrieve input
        for event in events:
            sparse_image = event['sparse_image']  # array of shape (Nx2) [points vs (1d pos, charge)]

            pos_ini = event['pos_ini']  # array with initial position (x1, y1, z1)
            pos_fin = event['pos_fin']
            proton_len = np.linalg.norm(pos_fin - pos_ini)
            ke = event['ke']  # Kinetic energy
            theta = event['theta']
            phi = event['phi']

            assert sparse_image.shape[0] > 0

            # reconstruct the image from sparse points to a 7x7x7 volume
            full_image = np.zeros(shape=(self.img_size * self.img_size * self.img_size))
            full_image[sparse_image[:, 0].astype(int)] = sparse_image[:, 1]
            full_image = full_image.reshape(7, 7, 7)

            # shift the image randomly towards a different
            # cube within a 3x3x3 sub-volume
            extended_image = np.zeros(shape=((self.img_size+2),
                                             (self.img_size+2),
                                             (self.img_size+2)))

            extended_image[ext_x:ext_x+self.img_size,
                           ext_y:ext_y+self.img_size,
                           ext_z:ext_z+self.img_size] = full_image

            pos_ini[0] += ((ext_x-1) * self.cube_size)
            pos_ini[1] += ((ext_y-1) * self.cube_size)
            pos_ini[2] += ((ext_z-1) * self.cube_size)

            # normalise
            extended_image = np.interp(extended_image.ravel(), (self.min_charge_new, self.max_charge_new),
                                       self.target_range).reshape(extended_image.shape)
            pos_ini = np.interp(pos_ini.ravel(), (self.min_pos, self.max_pos), self.source_range).reshape(pos_ini.shape)
            ke = np.interp(ke, (self.min_KE, self.max_KE), self.source_range).reshape(1)
            theta = np.interp(theta, (self.min_theta, self.max_theta), self.source_range).reshape(1)
            phi = np.interp(phi, (self.min_phi, self.max_phi), self.source_range).reshape(1)

            images.append(extended_image)
            params.append(np.concatenate((pos_ini, ke, theta, phi)))
            lens.append(proton_len)

            del event

        if len(images) == 0:
            return {'images': None,
                    'params': None,
                    }

        images = np.array(images)
        params = np.array(params)
        lens = np.array(lens)

        # sort protons by KE (descendent)
        order = params[:, 3].argsort()[::-1]
        params = params[order]
        lens = lens[order]

        return {'images': images,
                'ini_pos': params[:, :3].mean(axis=0),
                'params': params[:, 3:],
                'lens': lens,
               }