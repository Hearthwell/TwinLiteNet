from typing import Tuple

import torch
import torch.nn as nn

import tensorflow as tf
import numpy as np
import DataSet as myDataLoader

from argparse import ArgumentParser
from IOUEval import SegmentationMetric
from model.TwinLite import TwinLiteNet
from model.TwinLiteRELU import TwinLiteNet as TwinLiteRELU 
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def poly_lr_scheduler(args, optimizer, epoch, power=2):
    lr = round(args.lr * (1 - epoch / args.max_epochs) ** power, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def val(val_loader, model, runner, da_output_idx = 0):

    DA=SegmentationMetric(2)
    LL=SegmentationMetric(2)

    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()
    total_batches = len(val_loader)
    
    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=total_batches)
    for i, (_,input, target) in pbar:
        input = input.float() / 255.0

        input_var = input

        (out_da, out_ll) = runner(model, input_var, da_output_idx)

        target_da,target_ll=target

        _,da_predict=torch.max(out_da, 1)
        _,da_gt=torch.max(target_da, 1)

        _,ll_predict=torch.max(out_ll, 1)
        _,ll_gt=torch.max(target_ll, 1)

        DA.reset()
        DA.addBatch(da_predict.cpu(), da_gt.cpu())


        da_acc = DA.pixelAccuracy()
        da_IoU = DA.IntersectionOverUnion()
        da_mIoU = DA.meanIntersectionOverUnion()

        da_acc_seg.update(da_acc,input.size(0))
        da_IoU_seg.update(da_IoU,input.size(0))
        da_mIoU_seg.update(da_mIoU,input.size(0))


        LL.reset()
        LL.addBatch(ll_predict.cpu(), ll_gt.cpu())


        ll_acc = LL.pixelAccuracy()
        ll_IoU = LL.IntersectionOverUnion()
        ll_mIoU = LL.meanIntersectionOverUnion()

        ll_acc_seg.update(ll_acc,input.size(0))
        ll_IoU_seg.update(ll_IoU,input.size(0))
        ll_mIoU_seg.update(ll_mIoU,input.size(0))

    da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg,ll_IoU_seg.avg,ll_mIoU_seg.avg)
    return da_segment_result,ll_segment_result


def run_pytorch(model: torch.nn.Module, input_tensor: torch.Tensor, da_output_idx = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    return model(input_tensor)

def run_tflite(model: tf.lite.Interpreter, input_tensor: torch.Tensor, da_output_idx = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    ll_output_idx = 1 if da_output_idx == 0 else 0
    input_tensor = input_tensor.numpy()
    input_tensor = np.transpose(input_tensor, (0, 2, 3, 1))
    model.set_tensor(model.get_input_details()[0]['index'], input_tensor)
    model.invoke()

    da_output = model.get_tensor(model.get_output_details()[da_output_idx]['index'])
    da_output = np.transpose(da_output, (0, 3, 1, 2))
    da_output = torch.from_numpy(da_output)

    ll_output = model.get_tensor(model.get_output_details()[ll_output_idx]['index'])
    ll_output = np.transpose(ll_output, (0, 3, 1, 2))
    ll_output = torch.from_numpy(ll_output)
    return (da_output, ll_output)

def load_torch_model(state_dict) -> torch.nn.Module:
    try:
        model = TwinLiteNet()
        model.load_state_dict(state_dict)
        return model
    except:
        pass
    try:
        model = TwinLiteNet()
        # BECAUSE MODEL WAS WRAPPED INSIDE A MODULE FOR PARRALLEL TRAINING
        state_dict = { ".".join(k.split(".")[1:]) : v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        return model
    except:
        pass
    try:
        model = TwinLiteNet()
        model = nn.Sequential(torch.quantization.QuantStub(), model, torch.quantization.DeQuantStub())
        model.load_state_dict(state_dict)
        return model
    except:
        pass
    try:
        model = TwinLiteRELU()
        model.load_state_dict(state_dict)
        return model
    except:
        pass
    try:
        model = TwinLiteRELU()
        model = nn.Sequential(torch.quantization.QuantStub(), model, torch.quantization.DeQuantStub())
        model.load_state_dict(state_dict)
        return model
    except:
        raise ValueError("Model not recognized")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('model', default='', help='model path.')
    parser.add_argument('dataset_dir', type=str, help='Path to the dataset directory')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads to use for tflite interpreter')
    parser.add_argument('--da_idx', type=int, default=1, help='Index of DA output in tflite model')
    parser.add_argument('--num_image', type=int, default=0, help='Number of images to test')
    args = parser.parse_args()

    dataset = myDataLoader.CustomDataset(args.dataset_dir, valid=True)
    dataset.names = dataset.names[:args.num_image] if args.num_image > 0 else dataset.names
    valLoader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    model = None
    runner = None
    if args.model.endswith('.tflite'):
        runner = run_tflite
        model = tf.lite.Interpreter(model_path=args.model, num_threads=args.num_threads)
        model.allocate_tensors()
    elif args.model.endswith('.pth') or args.model.endswith('.pt'):
        runner = run_pytorch
        state_dict = torch.load(args.model, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model = load_torch_model(state_dict)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()

    da_segment_result, ll_segment_result = val(valLoader, model, runner, da_output_idx=args.da_idx)

    da_acc_seg, da_IoU_seg, da_mIoU_seg = da_segment_result
    ll_acc_seg, ll_IoU_seg, ll_mIoU_seg = ll_segment_result

    print(f"DA: {da_segment_result}")
    print(f"LL: {ll_segment_result}")