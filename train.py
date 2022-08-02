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
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from dataset import cocoDataset


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


def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, checkpoint='model.pth'):
    best_spearman = 0
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        model.train()
        for data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                reference = data['reference'].cuda()
                distorted = data['distorted'].cuda()
                labels = data['delta_target'].cuda().float()
                outputs = model(reference, distorted)
                loss = criterion(outputs, labels[:, None])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_loss = []
        outs_val = []
        gt_val = []
        with torch.no_grad():
            for data in tqdm.tqdm(val_loader):
                with torch.cuda.amp.autocast():
                    reference = data['reference'].cuda()
                    distorted = data['distorted'].cuda()
                    labels = data['delta_target'].cuda().float()
                    outputs = model(reference, distorted)
                    loss = criterion(outputs, labels[:, None])
                val_loss.append(loss.item())
                outs_val.extend(list(outputs[:, 0].detach().cpu().numpy()))
                gt_val.extend(list(labels.detach().cpu().numpy()))

        mean_loss = np.mean(val_loss)
        spearman = scipy.stats.spearmanr(outs_val, gt_val)[0]
        pearsonr = scipy.stats.pearsonr(outs_val, gt_val)[0]
        kendall = scipy.stats.kendalltau(outs_val, gt_val)[0]

        print(f"Valid Loss: {mean_loss:3f} | Spearman: {spearman:3f} | Pearson: {pearsonr:3f} | Kendall: {kendall:3f}")

        if spearman > best_spearman:
            best_spearman = spearman
            torch.save(model.state_dict(), checkpoint)



if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--identity-path', type=str, default='identity_CelebA.txt',
    #                                        help='Path to identity mappings')
    # parser.add_argument('--target-embeddings', type=str, default='arcface_embeddings.pickle',
    #                                        help='Target embeddings pickle file')
    # parser.add_argument('--target-codec-embeddings', type=str, default='arcface_codec_embeddings.pickle',
    #                                        help='Target codec embeddings pickle file')
    # parser.add_argument('--dataset-path', type=str, default='CelebA_align_v3_decoded',
    #                                        help='Dataset path')
    # parser.add_argument('--train-samples', type=int, default=5000,
    #                                        help='Number of random samples from training set')
    # parser.add_argument('--val-samples', type=int, default=2000,
    #                                        help='Number of random samples for validation')
    # parser.add_argument('--avg-samples', type=int, default=10,
    #                                        help='Average this number of samples for single distortion for more robust scoring')
    # parser.add_argument('--batch-size', type=int, default=32,
    #                                        help='Batch size for train and validation dataloaders')
    # parser.add_argument('--num-epochs', type=int, default=30,
    #                                        help='Number of epochs')
    # parser.add_argument('--checkpoint', type=str, default='model.pth',
    #                                        help='Path for checkpoint')
    # args = parser.parse_args()


    # identity = pd.read_csv(args.identity_path, sep=' ', header=None, names=['filename', 'id'])
    # identity['filename'] = identity['filename'].apply(lambda x: x[:-4])
    # emb_base = np.load(args.target_embeddings, allow_pickle=True)
    # emb_distortions = np.load(args.target_codec_embeddings, allow_pickle=True)
    # map_base_name_to_ref = {}
    # ref_emb_all = set([x[:-4] for x in emb_distortions['ref'].keys()])
    # for emb_name_base in tqdm.tqdm(emb_base.keys()):
    #     person_id = int(identity[identity['filename'] == emb_name_base[:-4]]['id'])
    #     mapped_ref = list(set(identity[identity['id'] == person_id]['filename']).intersection(ref_emb_all))
    #     assert len(mapped_ref) == 1
    #     map_base_name_to_ref[emb_name_base[:-4]] = mapped_ref[0]
    # celeba_distortions_path = args.dataset_path


    # X_train, X_test = train_test_split(sorted([x[:-4] for x in emb_base.keys()]), test_size=0.3, random_state=1337)
    # NUM_SAMPLES = args.avg_samples
    # data_transform = torchvision.transforms.Compose([
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                          std=[0.229, 0.224, 0.225])])

    # dataset_train = StochasticDataset(args.train_samples, NUM_SAMPLES, celeba_distortions_path, emb_base,
    #                     emb_distortions, map_base_name_to_ref, X_train, is_val=False, transform=data_transform)
    # dataset_val = StochasticDataset(args.val_samples, NUM_SAMPLES, celeba_distortions_path, emb_base,
    #                     emb_distortions, map_base_name_to_ref, X_test, is_val=True, transform=data_transform)

    batch_size = 1

    train_dataset = cocoDataset()
    val_dataset = cocoDataset(validation=True)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                   pin_memory=False, shuffle=True, num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size,
                                 pin_memory=False, shuffle=False, num_workers=0, drop_last=True)

    criterion_l1 = nn.L1Loss()
    model = Net().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    train(model, optimizer, criterion_l1, train_loader, val_loader, 1, 'model.pth')