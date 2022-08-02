import os
import time
import torch
import numpy as np
from PIL import Image
import sklearn.metrics
from torch.utils.data import Dataset, DataLoader

class StochasticDataset(torch.utils.data.IterableDataset):
    
    def __init__(self, size, num_samples, celeba_distortions_path, emb_base, 
                 emb_distortions, map_base_name_to_ref, file_names, is_val=False, transform=None):
        self.celeba_distortions_path = celeba_distortions_path
        self.emb_base = emb_base
        self.emb_distortions = emb_distortions
        self.distortions_folders = sorted(os.listdir(celeba_distortions_path))
        self.file_names = sorted(file_names)
        self.transform = transform
        self.map_base_name_to_ref = map_base_name_to_ref
        self.num_samples = num_samples
        self.dist = sorted(os.listdir(self.celeba_distortions_path))
        self.size = size
        self.is_val = is_val
        
    def __len__(self):
        return self.size
    
    def __iter__(self):
        
        for i in range(self.size):
            
            if self.is_val:
                np.random.seed(i)
            else: 
                np.random.seed(int(time.time()))
            
            list_distorted = []
            list_reference = []
            list_deltas = []
        
            distortion_name = np.random.choice(self.dist)
            
            for num_tryes in range(self.num_samples):

                base_name = np.random.choice(self.file_names)

                ref_name = self.map_base_name_to_ref[base_name]

                if 'jpeg' in distortion_name:
                    img_distorted = os.path.join(self.celeba_distortions_path, distortion_name, ref_name + '.jpg')
                else:
                    img_distorted = os.path.join(self.celeba_distortions_path, distortion_name, ref_name + '.png')

                img_reference = os.path.join(self.celeba_distortions_path, 'ref', ref_name + '.png')

                distorted = self.transform(Image.open(img_distorted).convert('RGB'))
                reference = self.transform(Image.open(img_reference).convert('RGB'))

                if 'jpeg' in distortion_name:
                    emb_distorted = self.emb_distortions[distortion_name][ref_name + '.jpg']
                else:
                    emb_distorted = self.emb_distortions[distortion_name][ref_name + '.png']

                emb_ref = self.emb_distortions['ref'][ref_name + '.png']
                emb_in_base = self.emb_base[base_name + '.png']

                a = sklearn.metrics.pairwise.cosine_similarity([emb_ref], [emb_in_base])[0][0]
                b = sklearn.metrics.pairwise.cosine_similarity([emb_distorted], [emb_in_base])[0][0]
                
                list_distorted.append(distorted)
                list_reference.append(reference)
                list_deltas.append(a-b)
                
            yield {
                    'distorted':torch.stack(list_distorted), 
                    'reference':torch.stack(list_reference), 
                    'delta_with_hidden_target': np.mean(list_deltas)
                  }