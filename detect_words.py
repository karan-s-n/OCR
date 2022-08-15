
"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import json
import zipfile
import yaml 
from PIL import Image
import cv2
from skimage import io
import numpy as np
from collections import OrderedDict

#userdefined packages 
from utils import utils as ut
import utils.imageutils as im
from utils.architect import CRAFT

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


class Configuration:
  def __init__(self):
    with open('config.yml',"r") as f:
        self.config = yaml.load(f, Loader=yaml.Loader)

  def get_attrib(self):
    self.trained_model  = self.config["trained_model"]
    self.text_threshold  = self.config["text_threshold"]
    self.low_text  = self.config["low_text"]
    self.link_threshold  = self.config["link_threshold"]
    self.cuda  = self.config["cuda"]
    self.canvas_size = self.config["canvas_size"]
    self.mag_ratio =  self.config["mag_ratio"]
    self.poly = self.config["poly"]
    self.show_time = self.config["show_time"]
    self.test_folder = self.config["test_folder"]
    self.report_folder = self.config["report_folder"]
    self.data_limit = self.config["data_limit"]
  

args = Configuration()
args.get_attrib()

""" For test images in a folder """
image_list, _, _ = ut.get_files(args.test_folder)

result_folder = args.report_folder
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = im.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = im.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    
    

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = ut.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    print("no. of boxes detected:",len(boxes))
    # coordinate adjustment
    boxes = ut.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = ut.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: 
          polys[k] = boxes[k]
    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = im.cvt2HeatmapImg(render_img)
    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
    return boxes, polys, ret_score_text



if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize
    refine_net = None

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    net.eval()
    t = time.time()
    image_list = image_list[:args.data_limit]
    # load data
    import pandas as pd
    dataset = pd.DataFrame()
    for k, image_path in enumerate(image_list):
        image = im.loadImage(image_path)
        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)
        strResult = ""
        for box in  polys:
          rect = cv2.boundingRect(box)
          x,y,w,h = rect
          dataset = dataset.append({"image_path":image_path , "x":int(x),"y":int(y),"w":int(w),"h":int(h)},ignore_index = True)
        ut.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)
    dataset.to_csv(result_folder+"/dataset.csv",index=False)
    print("elapsed time : {}s".format(time.time() - t))
