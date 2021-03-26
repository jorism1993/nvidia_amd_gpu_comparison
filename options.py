import os
import argparse
import torch
import time
import random


def get_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset options
    parser.add_argument('--num_images_train', default=1e5, type=int, help='Number of images per dataset')

    # Basic training options
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs during training')
    parser.add_argument('--skip_batches', type=int, default=5, help='Number of batches to skip for logging')
    parser.add_argument('--skip_epoch', type=int, default=1, help='Number of epochs to skip for logging')

    # Model and image options
    parser.add_argument('--model_type', type=str, default='resnet101', help='Type of model to use')
    parser.add_argument('--img_size', type=int, default=299, help='Input image size')

    # Optimizer, learning rate (scheduler) and early stopping options
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon for Adam optimizer')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for gradient updates')

    # Cuda options
    parser.add_argument('--no_cuda', action='store_true', help='Use this to train without cuda enabled')
    parser.add_argument('--cuda_devices', nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')

    # Logging options
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output files to')

    # Misc
    parser.add_argument('--seed', type=int, default=random.randint(0, 99999), help='Random seed to use')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for dataloader')

    opts = parser.parse_args()

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.cuda_devices = list(sorted([int(i) for i in opts.cuda_devices]))
    opts.device = f"cuda:{opts.cuda_devices[0]}" if opts.use_cuda else "cpu"

    if not opts.device == 'cpu':
        torch.cuda.set_device(opts.device)

    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.output_dir = os.path.join(opts.output_dir, opts.model_type, opts.run_name)
    os.makedirs(opts.output_dir, exist_ok=True)

    return opts