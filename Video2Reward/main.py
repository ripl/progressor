import datetime
import numpy as np
import os
import time,tqdm,random
from pathlib import Path
import wandb,pdb,cv2

import torch, torchvision
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch import distributions
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from einops import rearrange
from PIL import Image
import matplotlib.pyplot as plt
import torch.distributed as dist

from util.frame_triplet_loader import Frameloader, SegmentFrameloader
from util.transforms import RandomApplyTransform, RandomShiftsAug, GaussianNoise
from util.util import ddp_setup, save_checkpoint, load_checkpoint, get_args_parser
from reward_model import Model
import util.misc as misc
mean=np.array([0.485, 0.456, 0.406])
std=np.array([0.229, 0.224, 0.225])
@torch.no_grad()
def eval_model(device_id, model, test_loader, minibatch = 16, save_path = 'img.png'):
    model.eval()
    out_list = []
    for idx, (init_img, goal_img, mid_img, relative_position, delta_goal_init) in enumerate(tqdm.tqdm(test_loader)):
        seq_list = []
        init_img = init_img[None,...].to(device_id)
        goal_img = goal_img[None,...].to(device_id)
        mid_img = mid_img.to(device_id)
        feat_list = []
        for idx in range((mid_img.shape[0] + minibatch - 1) // minibatch):
            mid_img_batch = mid_img[idx * minibatch: (idx + 1) * minibatch]
            out = model([init_img.expand(mid_img_batch.shape[0],-1,-1,-1), mid_img_batch,
                  goal_img.expand(mid_img_batch.shape[0],-1,-1,-1)], return_feat = False)
            seq_list.append(out.loc)
            #feat_list.append(feat.reshape(-1,3,feat.shape[-1]))
        seq = torch.cat(seq_list)
        seq = torch.nn.functional.interpolate(seq.reshape(1,1,-1), size = (100), mode = 'linear').reshape(-1)
        #feat = torch.cat(feat_list)
        #print(compute_logdet(feat[:,1].reshape(-1,128)))
        #seq = seq - seq[0]
        #seq = seq * 1 / seq[-1]
        out_list.append(seq)
    seq = torch.stack(out_list)
    seq_mean = seq.mean(dim = 0).cpu().data.numpy()
    fig, ax = plt.subplots()
    ax.plot(np.arange(100), seq_mean, linewidth = 4.0)
    ax.fill_between(np.arange(100), seq_mean - seq.std(dim = 0).cpu().data.numpy(), seq_mean + seq.std(dim = 0).cpu().data.numpy(), color='#888888', alpha=0.4)
    plt.savefig(save_path)
    plt.close()
    model.train()
@torch.no_grad()
def plot_eval_model(device_id, model, test_loader, minibatch = 16, save_path = 'img.png'):
    model.eval()
    out_list = []
    def compose_img_plot_list(goal_img_list, mid_img_list, left_img_list, plot_img_list):
        assert len(goal_img_list) == len(mid_img_list)
        assert len(mid_img_list) == len(left_img_list)
        assert len(left_img_list) == len(plot_img_list)
        final_img_list = []
        for goal_img, mid_img, left_img, plot_img in zip(goal_img_list, mid_img_list, left_img_list, plot_img_list):
            resized_image = cv2.resize(plot_img, (224, 224))
            all_img = np.zeros((448, 224 * 3, 3))
            all_img[0:224, 224:224 * 2] = resized_image
            all_img[224:448, :224] = goal_img
            all_img[224:448, 224:224 * 2] = mid_img
            all_img[224:448, 224 * 2: 224 * 3] = left_img
            final_img_list.append(all_img.astype(np.uint8)[:,:,::-1])
        return final_img_list

    def normalize_img_array_to_list(img_array):
        img_array = img_array.permute(0,2,3,1).cpu().data
        num_of_img = img_array.shape[0]
        return [np.clip(np.array(img), a_min = 0, a_max = 255).astype(np.uint8) for img in ((img_array * std.reshape(1,1,3) + mean.reshape(1,1,3)) * 255).chunk(num_of_img, dim = 0)]

    def render_video(img_list, video_name, fps = 1):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = img_list[0].shape[:2]
        video = cv2.VideoWriter(video_name + '.avi', fourcc, fps, (width, height))
        for img in img_list:
            video.write(img)
        video.release()

    def draw_plot(mu, sigma):
        mu = mu.reshape(-1).cpu().data.numpy()
        sigma = sigma.reshape(-1).cpu().data.numpy()
        xaxis_limit = mu.shape[0]
        data_list = []
        for i in range(1, xaxis_limit):
            fig = plt.figure(figsize = (4,4))
            fig.add_subplot(111)
            this_mu = mu[:i]
            this_sigma = sigma[:i]
            plt.plot(np.arange(this_mu.shape[0]), this_mu)
            plt.fill_between(np.arange(this_mu.shape[0]), this_mu - this_sigma, this_mu + this_sigma, color='#8FAADC', alpha=0.6)
            ax = plt.gca()
            ax.set_xlim([0, xaxis_limit + 1])
            ax.set_ylim([-0.3, 1.3])
            fig.tight_layout()
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            data_list.append(data)
            plt.close()
        return data_list
    for idx, (init_img, goal_img, mid_img, relative_position, delta_goal_init) in enumerate(tqdm.tqdm(test_loader)):
        seq_list = []
        sigma_list = []
        init_img = init_img[None,...].to(device_id)
        goal_img = goal_img[None,...].to(device_id)
        mid_img = mid_img.to(device_id)
        xaxis_range = mid_img.shape[0]
        for idx in range((mid_img.shape[0] + minibatch - 1) // minibatch):
            mid_img_batch = mid_img[idx * minibatch: (idx + 1) * minibatch]
            out = model([init_img.expand(mid_img_batch.shape[0],-1,-1,-1), mid_img_batch,
                  goal_img.expand(mid_img_batch.shape[0],-1,-1,-1)])
            seq_list.append(out.loc)
            sigma_list.append(out.scale)
        plot_img_list = draw_plot(torch.cat(seq_list), torch.cat(sigma_list))
        mid_img_list = normalize_img_array_to_list(mid_img)[:len(plot_img_list)]
        init_img_list = normalize_img_array_to_list(init_img) * len(mid_img_list)
        goal_img_list = normalize_img_array_to_list(goal_img) * len(mid_img_list)
        render_video(compose_img_plot_list(init_img_list, mid_img_list, goal_img_list, plot_img_list), save_path, 60)
        return
def print_gradients(model):
    norms = []
    max_grad = 0
    max_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            max_grad = max(max_grad, p.grad.norm())
            max_norm = max(max_norm, p.data.norm(2))
    return max_grad, max_norm

def img_unnormalize(img, row = 1, col = 1):
    img_h, img_w = img.shape[2:]
    img = (((img * 0.5 + 0.5) * 255).cpu().data.numpy()).transpose(0,2,3,1).astype(np.uint8)
    img = img[:row * col].reshape(row, col, img_h, img_w, 3).transpose(0,2, 1, 3, 4)
    img = img.reshape(row * img_h, col * img_w, 3)
    return img

def adjust_learning_rate(optimizer, args, iter, iter_per_epoch):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if iter < args.warmup_epochs * iter_per_epoch:
        lr = max(args.lr * iter / (args.warmup_epochs * iter_per_epoch), 1e-7)
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
from PIL import Image, ImageFilter, ImageOps
class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def compute_logdet(W, eps = 0.2):
    m, p = W.shape  # [d, B]
    W = W.float()
    I = torch.eye(p, device=W.device)
    scalar = p / (m * eps)
    cov = W.T.matmul(W)
    cov = torch.stack(FullGatherLayer.apply(cov)).mean(dim = 0)
    logdet = torch.logdet(I + scalar * cov)
    return logdet / 2.
class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Login to wandb for logging
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
        wandb.init(project='video2reward', group='ddp')

    # Image augmentation: color jitter, gaussian noise, random shifts

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    geo_augment = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip()
            ])
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
        transforms.ToTensor(),
        normalize
    ])

    transform_test= transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        normalize
    ])

    # Initialize dataset
    dataset_train = Frameloader(
        root_dir=args.data_path,
        train = True,
        train_test_split_ratio = 0.8,
        transforms=transform_train,
        geo_augment=geo_augment,
        normalize_trajectory=args.normalize_prediction,
        subset_len=1,
        max_frame_gap = 2000,
        randomized=args.randomize,
        sample_vip_frame = True
        )
    dataset_train[0]
    dataset_eval_syn = SegmentFrameloader(
        root_dir='/path/to/data',  # TODO: replace with actual path
        train = True,
        train_test_split_ratio = 1.0,
        transforms=transform_test,
        normalize_trajectory=True,
        segment_size = 100,
        frame_internal =1
        )
    dataset_eval_syn[0]
    dataset_eval= SegmentFrameloader(
        root_dir=args.data_path,
        train = False,
        train_test_split_ratio = 0.8,
        transforms=transform_test,
        normalize_trajectory=True,
        segment_size = 500,
        frame_internal =5
        )
    device_id = int(os.environ['LOCAL_RANK'])
    # Initialize sampler for distributed training
    if args.distributed:
        ddp_setup()
        misc.setup_for_distributed(device_id == 0)
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
        print('Sampler_train = %s' % str(sampler_train))
    else:
        print('single gpu mode')
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # Initialize dataloader
    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers = True
    )
    # Initialize model and optimizer
    model = Model(model_type='resnet18', latent_dim=512)
    model.to(device_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # Load checkpoint if resuming training
    if args.resume == 'last' or len(args.resume) > 0:
        model, optimizer, epoch  = load_checkpoint(args.output_dir, model, optimizer, args.resume)
        args.start_epoch = epoch
        print(f'resume from epoch {epoch}')
    # Initialize distributed model
    if args.distributed:
        model = DDP(model, device_ids=[device_id], broadcast_buffers=False)

    # Training loop
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        data_loader_train.sampler.set_epoch(epoch)

        # Batch loop
        epoch_loss = []
        epoch_mean_error = []

        metric_logger = misc.MetricLogger(delimiter="  ")
        header = 'Epoch: [{}]'.format(epoch)
        slot_vect_list = []
        print_freq = 10
        #for data_iter_step, feat_list in enumerate(metric_logger.log_every(loader, print_freq, header)):
        for idx, (cur_batch, cur_lbl, delta) in enumerate(metric_logger.log_every(data_loader_train, print_freq, header)):
            lr = adjust_learning_rate(optimizer, args, idx + epoch * len(data_loader_train), len(data_loader_train))
            cur_batch, cur_lbl, delta = cur_batch.to(device_id), cur_lbl.to(device_id), delta.to(device_id)
            # Dimensions: (batch_size, num_frames, channels, height, width)
            # True batch size is batch_size * num_frames

            cur_batch = rearrange(cur_batch, 'b i c h w -> (b i) c h w')
            cur_lbl = rearrange(cur_lbl, 'b i -> (b i)')
            delta = rearrange(delta,'b i -> (b i)')
            optimizer.zero_grad()
            img_list = cur_batch.chunk(3, dim = 1)
            
            # Predicted distribution
            pred_dist, feat, feat_norm = model(img_list)
            logdet = compute_logdet(feat)
            neg_img_list = []
            neg_img_list.append(img_list[0].detach())
            neg_img_list.append(torch.roll(img_list[1], 1, dims = 0).detach())
            neg_img_list.append(img_list[2].detach())
            pred_neg_dist, _,_ = model(neg_img_list)
            # Ground truth distribution
           
            gt_std = 0.1
            gt_mean = cur_lbl.unsqueeze(-1)
            lbl = distributions.Normal(gt_mean, gt_std)
            neg_lbl = distributions.Normal(torch.ones_like(gt_mean) * -1, gt_std)
            # Loss calculation
            #loss = 100 * torch.nn.functional.mse_loss(gt_mean.reshape(-1).float(), pred_dist.loc.reshape(-1).float())
            loss = distributions.kl.kl_divergence(lbl, pred_dist).sum(dim = -1).mean()
            neg_loss = distributions.kl.kl_divergence(neg_lbl, pred_neg_dist).sum(dim = -1).mean()
            mean_error = (torch.abs(pred_dist.loc- gt_mean)).mean().item()
            epoch_loss.append(loss.item())
            epoch_mean_error.append(mean_error)
            metric_logger.update(loss=loss.item())
            metric_logger.update(feat_norm=feat_norm.mean().item())
            metric_logger.update(logdet=logdet.item())
            metric_logger.update(neg_loss=neg_loss.item())
            #metric_logger.update(lr=lr)
            metric_logger.update(mean=mean_error)
            #metric_logger.update(gt_std=gt_std.mean())
            metric_logger.update(pred_std=pred_dist.loc.reshape(-1).std())
            #metric_logger.update(gt_std_min=gt_std.min())
            (loss + 0.05 * neg_loss).backward()
            max_grad, max_norm = print_gradients(model)
            metric_logger.update(max_grad =max_grad)
            metric_logger.update(max_norm=max_norm)
            optimizer.step()

        # Evaluation loop
        if args.eval_path != '' and (epoch % args.eval_every == 0 or epoch == args.epochs - 1):
            if device_id == 0:
                eval_model(device_id, model, dataset_eval, minibatch = 16, save_path = args.output_dir + f'/eval_{epoch:07d}.png')
                eval_model(device_id, model, dataset_eval_syn, minibatch = 16, save_path = args.output_dir + f'/sync_eval_{epoch:07d}.png')
            torch.distributed.barrier()
        # If not evaluating, log only training metrics to wandb
        elif args.wandb_key:
            wandb.log({'Loss': np.array(epoch_loss).mean(),
                       'Mean Error': np.array(epoch_mean_error).mean()})

        # Update learning rate
        #scheduler.step()

        # Save model checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            if args.distributed:
                if device_id == 0:
                    save_checkpoint(model.module, epoch, args.output_dir, optimizer)
            else:
                save_checkpoint(model, epoch, args.output_dir, optimizer)

    # Print total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    destroy_process_group()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
