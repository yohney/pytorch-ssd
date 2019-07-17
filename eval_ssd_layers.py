import torch
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.ssd import MatchPrior
from vision.ssd.layer_descriptor import LayerApCalculator
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
import argparse
import pathlib
import pandas as pd
import numpy as np
import time
import logging
import sys
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.data_preprocessing import TestTransform

parser = argparse.ArgumentParser(description="SSD Evaluation per layer - writing down prior boy accuracy on each layer.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str)

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if __name__ == '__main__':
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]

    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True, convert_to_boxes=False)
        config = vgg_ssd_config
    elif args.net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True, convert_to_boxes=False)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True, convert_to_boxes=False)
        print("Not defined config for mb1 lite")
        exit(-1)
    elif args.net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True, convert_to_boxes=False)
        config = squeezenet_ssd_config
    elif args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True, convert_to_boxes=False)
        print("Not defined config for mb2 lite")
        exit(-1)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)  

    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    if args.dataset_type == "voc":
        dataset = VOCDataset(args.dataset, is_test=True, transform=test_transform, target_transform=target_transform)
    elif args.dataset_type == 'open_images':
        dataset = OpenImagesDataset(args.dataset, dataset_type="test", transform=test_transform, target_transform=target_transform)

    timer.start("Load Model")
    net.load(args.trained_model)
    net = net.to(DEVICE)
    net.eval()
    print(f'It took {timer.end("Load Model")} seconds to load the model.')

    apCalculator = LayerApCalculator(net, title="Evaluation Layer Stats")
  
    results = []
    
    start = time.time()
    for i in range(len(dataset)):
        #print("process image", i)
        timer.start("Load Image")
        image, boxes, labels = dataset.__getitem__(i)
        #print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.start("Predict")
        apCalculator.update_layer_stats(image, labels)

        report_freq = 5
        if i > 0 and i % report_freq == 0:
            print("{} / {}".format(i, len(dataset)))
            print("ETA: {:.2f} min".format((time.time() - start) / 60 / report_freq * (len(dataset)-i)))
            print("")
            start = time.time()
            break

    apCalculator.print()


