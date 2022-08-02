import os
import cv2
import tqdm
import torch
import scipy
import pickle
import argparse
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from dataset import StochasticDataset
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, 1)
        
    def forward(self, ref, dist):
        b, s, c, h, w = ref.shape
        a = self.backbone(ref.view((-1, c, h, w)))
        b = self.backbone(dist.view((-1, c, h, w)))
        diff = (a - b).view((-1, s, 512))
        pooled = diff.mean(dim=1)
        x = self.fc(pooled)
        return x


def eval(model, loader):
    scaler = torch.cuda.amp.GradScaler()
    model.eval()
    outs_val = []
    gt_val = []
    with torch.no_grad():
        for data in tqdm.tqdm(loader):
            with torch.cuda.amp.autocast():
                reference = data['reference'].cuda()
                distorted = data['distorted'].cuda()
                labels = data['delta_with_hidden_target'].cuda().float()
                outputs = model(reference, distorted)
                outs_val.extend(list(outputs[:, 0].detach().cpu().numpy()))
                gt_val.extend(list(labels.detach().cpu().numpy()))

    spearman = scipy.stats.spearmanr(outs_val, gt_val)[0]
    pearsonr = scipy.stats.pearsonr(outs_val, gt_val)[0]

    print(f"Spearman: {spearman:3f} | Pearson: {pearsonr:3f}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Evaluate model on dataset and calculate correlations')
    parser.add_argument('--identity-path', type=str, default='identity_CelebA.txt',
                                           help='Path to identity mappings')
    parser.add_argument('--target-embeddings', type=str, default='arcface_embeddings.pickle',
                                           help='Target embeddings pickle file')
    parser.add_argument('--target-codec-embeddings', type=str, default='arcface_codec_embeddings.pickle',
                                           help='Target codec embeddings pickle file')
    parser.add_argument('--dataset-path', type=str, default='CelebA_align_v3_decoded',
                                           help='Dataset path')
    parser.add_argument('--test-samples', type=int, default=10000,
                                           help='Number of random samples for testing (not greather than C(len(dataset), #samples))')
    parser.add_argument('--avg-samples', type=int, default=10,
                                           help='Average this number of samples for single distortion for more robust scoring')
    parser.add_argument('--batch-size', type=int, default=32,
                                           help='Batch size for dataloader')
    parser.add_argument('--checkpoint', type=str, default='model.pth',
                                           help='Path for checkpoint')
    args = parser.parse_args()


    identity = pd.read_csv(args.identity_path, sep=' ', header=None, names=['filename', 'id'])
    identity['filename'] = identity['filename'].apply(lambda x: x[:-4])
    emb_base = np.load(args.target_embeddings, allow_pickle=True)
    emb_distortions = np.load(args.target_codec_embeddings, allow_pickle=True)
    map_base_name_to_ref = {}
    ref_emb_all = set([x[:-4] for x in emb_distortions['ref'].keys()])
    for emb_name_base in tqdm.tqdm(emb_base.keys()):
        person_id = int(identity[identity['filename'] == emb_name_base[:-4]]['id'])
        mapped_ref = list(set(identity[identity['id'] == person_id]['filename']).intersection(ref_emb_all))
        assert len(mapped_ref) == 1
        map_base_name_to_ref[emb_name_base[:-4]] = mapped_ref[0]
    celeba_distortions_path = args.dataset_path


    X_train, X_test = train_test_split(sorted([x[:-4] for x in emb_base.keys()]), test_size=0.3, random_state=1337)
    NUM_SAMPLES = args.avg_samples
    data_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

    dataset = StochasticDataset(args.test_samples, NUM_SAMPLES, celeba_distortions_path, emb_base, 
                        emb_distortions, map_base_name_to_ref, X_test, is_val=True, transform=data_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)
    model = Net().cuda().eval()
    model.load_state_dict(torch.load(args.checkpoint))
    eval(model, loader)