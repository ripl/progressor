import torch
import torch.nn as nn
import torchvision,pdb
import numpy as np
from torch import distributions
#from vit import mae_vit_base_patch16
F = torch.nn.functional
#import resnet

class MLP(nn.Module):
    ''' MLP as used in Vision Transformer, MLP-Mixer and related networks.
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or hidden_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features)
        else:
            self.skip = nn.Identity()

    def forward(self, x_input):
        x = self.fc1(x_input)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x + self.skip(x_input)

def get_smoothed_variance(var_unconstrained):
    return (1/0.693) * torch.log(1 + torch.exp(0.693 * var_unconstrained))

class Model(nn.Module):
    ''' Reward model; takes in a stack of 3 images and outputs a distribution over rewards.
    '''
    def __init__(self, latent_dim=512, model_type = 'resnet34'):
        super().__init__()
        #norm = lambda x: nn.GroupNorm(32, x)
        self.model = torchvision.models.resnet34(pretrained = False)
        self.model.fc = nn.Sequential(
                                     nn.Linear(512, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 128),
                                     )
        self.proj = nn.Sequential(MLP(3 * 128, 2048),
                                  MLP(2048, latent_dim//2))
        self.fc_mu = nn.Linear(latent_dim//2, 1)
        self.fc_log_var = nn.Linear(latent_dim//2, 1)

    @torch.no_grad()
    def compute_reward(self,init_mid_goal_img_stack):
        pred_reward_dist = self.forward(init_mid_goal_img_stack)
        if isinstance(pred_reward_dist, tuple):
            pred_reward_dist = pred_reward_dist[0]
        alpha = 0.4
        pred_reward = (pred_reward_dist.mean - alpha * pred_reward_dist.entropy()).item()
        return pred_reward

    def forward(self, img_list, return_feat = False):
        if isinstance(img_list, torch.Tensor):
            assert img_list.shape[1] == 9
            img_list = img_list.chunk(3, dim = 1)
        feat_list = []
        for img in img_list:
            feat_list.append(F.normalize(self.model(img), dim = -1))
        #feat_norm = ((feat_list[2] - feat_list[0]).norm(dim = -1, keepdim = True) + 1e-5)
        #feat = (feat_list[1] - feat_list[0]) / ((feat_list[2] - feat_list[0]).norm(dim = -1, keepdim = True) + 1e-5)
        #feat = torch.cat([feat_list[1] - feat_list[0], feat_list[2] - feat_list[0]], dim = 1)
        feat = torch.cat(feat_list, dim = -1)
        pred = self.proj(feat * np.sqrt(256))
        mu = self.fc_mu(pred)
        log_var = self.fc_log_var(pred)
        std = get_smoothed_variance(log_var)
        if self.training or return_feat:
            return distributions.Normal(mu, std), torch.stack(feat_list, dim = 0).flatten(0,1)
        else:
            return distributions.Normal(mu, std)

    def get_reward(self, img_list):
        pred_dist = self.forward(img_list)
        return pred_dist.mean.item() - 0.4 * pred_dist.entropy().item()
