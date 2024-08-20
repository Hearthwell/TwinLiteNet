import os
import torch
import torch.nn as nn
from model import TwinLite as net
from model.TwinLiteRELU import TwinLiteNet as TwinLiteRELU
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
import torch.optim.lr_scheduler

from loss import TotalLoss

def train_net(args):
    # load the model
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    num_gpus = torch.cuda.device_count()

    model = net.TwinLiteNet()

    if args.relu:
        model = TwinLiteRELU()
    
    if args.qat:
        model = nn.Sequential(torch.quantization.QuantStub(), 
                  model, 
                  torch.quantization.DeQuantStub())
        
        model.train()
        
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    args.savedir = args.savedir + '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)


    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.CustomDataset(args.dataset_dir),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


    valLoader = torch.utils.data.DataLoader(
        myDataLoader.CustomDataset(args.dataset_dir, valid=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    args.onGPU = False
    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    criteria = TotalLoss()

    start_epoch = 0
    lr = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained weights from '{}'".format(args.pretrained))
            pretrained_dict = torch.load(args.pretrained, map_location=device)
            if num_gpus <= 1:
                pretrained_dict = {".".join(k.split(".")[1:]): v for k, v in pretrained_dict.items()}
            # WE WANT TO SEE IF WE ARE MISSING SOME KEYS
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            print("=> no pretrained weights found at '{}'".format(args.pretrained))

    for epoch in range(start_epoch, args.max_epochs):

        model_file_name = args.savedir + os.sep + 'model_{}.pth'.format(epoch)
        poly_lr_scheduler(args, optimizer, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " +  str(lr))

        # train for one epoch
        model.train()
        train( args, trainLoader, model, criteria, optimizer, epoch)
        model.eval()
        # validation
        val(valLoader, model)
        torch.save(model.state_dict(), model_file_name)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr
        }, args.savedir + 'checkpoint.pth.tar')




if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('dataset_dir', type=str, help='Path to the dataset directory')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--savedir', default='./test_', help='directory to save the results')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')
    parser.add_argument('--pretrained', default='', help='Pretrained ESPNetv2 weights.')
    parser.add_argument('--qat', help='Quantization Aware Training', action='store_true')
    parser.add_argument('--relu', help='Replace PRELU activations with RELU', action='store_true')

    train_net(parser.parse_args())

