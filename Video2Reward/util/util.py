import argparse, os, glob
import torch
from torch.distributed import init_process_group


def get_args_parser():
    parser = argparse.ArgumentParser(prog='video2reward', description='')
    parser.add_argument('--input_size', default=84, type=int,
                        help='images input size')
    parser.add_argument('--normalize_prediction', action='store_true',
                        help='normalize progress prediction')
    parser.add_argument('--no_normalize_prediction', action='store_false', dest='normalize_prediction')
    parser.set_defaults(normalize_prediction=True)
    parser.add_argument('--augment', action='store_true',
                        help='apply image augmentation')
    parser.add_argument('--randomize', action='store_true',
                        help='randomly sample start and end frames')
    parser.add_argument('--no_randomize', action='store_false', dest='randomize')
    parser.set_defaults(randomize=True)
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--distributed', action='store_true',
                        help='distributed across nodes')
    parser.add_argument('--no_distributed', action='store_false', dest='distributed')
    parser.set_defaults(distributed=True)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='batch size')
    parser.add_argument('--data_path', default='../rank2reward', type=str,
                        help='dataset path')
    parser.add_argument('--eval_path', default='', type=str,
                        help='path to validation set')
    parser.add_argument('--eval_every', default=50, type=int,
                        help='frequency of model evaluation on the validation set')
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help='frequency of model evaluation on the validation set')
    parser.add_argument('--save_every', default=50, type=int,
                        help='frequency of saving weights')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of epoch')
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--wandb_key', default=None)
    return parser


def ddp_setup():
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def save_checkpoint(model, epoch, path, optimizer, save_last = False, scaler = None):
    if save_last:
        model_path = os.path.join(path, 'model_last.pth')
    else:
        model_path = os.path.join(path, 'model_' + str(epoch) + '.pth')

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'scaler_state_dict': scaler.state_dict() if scaler is not None else 0,
            },model_path)

def load_checkpoint(path, model, optimizer, epoch='last', scaler = None):
    if epoch == 'last':
        if os.path.exists(os.path.join(path, 'model_last.pth')):
            weight_path = os.path.join(path, 'model_last.pth')
        else:
            weight_list = glob.glob(path + '*.pth')
            last_epoch = max([int(m.split('_')[-1].split('.')[0]) for m in weight_list])
            weight_path = os.path.join(path, 'model_'+ str(last_epoch) + '.pth')
    else:
        weight_path = os.path.join(path, 'model_'+ str(epoch) + '.pth')
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return model, optimizer, epoch, scaler
    else:
        return model, optimizer, epoch
