import os
import argparse
import time
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys

import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import einops
from einops import rearrange
import sklearn.metrics
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from monodepth2.networks import ResnetEncoder, DepthDecoder
import PIL.Image as pil
sys.path.append('/nethome/abati7/flash/Work/jat/gaze360/code/')
from model import GazeLSTM

deviceDepth = "cuda:0"
deviceAngle = [1,2]

######################### Depth Estimator ###############################
image_path = "/nethome/abati7/flash/Data/gazefollow_extended/"
test = pd.read_csv("/nethome/abati7/flash/Data/test_annotations_release_angles.txt", sep=",")
train = pd.read_csv("/nethome/abati7/flash/Data/train_annotations_release_angles.txt", sep=",")

model_path = "/nethome/abati7/flash/Work/jat/"
encoder_path = os.path.join(model_path, "encoder.pth")
depth_decoder_path = os.path.join(model_path, "depth.pth")

encoder = ResnetEncoder(18, True)
loaded_dict_enc = torch.load(encoder_path, map_location="cpu")
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(deviceDepth)
encoder.eval()

depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
loaded_dict = torch.load(depth_decoder_path, map_location="cpu")
depth_decoder.load_state_dict(loaded_dict)
depth_decoder.to(deviceDepth)
depth_decoder.eval()

feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']

######################### 3D Gaze Direction ###############################
model = GazeLSTM()
model = nn.DataParallel(model, device_ids=deviceAngle)
checkpoint = torch.load('models/gaze360_model.pth.tar', map_location="cpu")
model.load_state_dict(checkpoint['state_dict'])
model.to("cuda:1")
model.eval()


def depthImageLoader(path):
    # testImage = "test2/00000003/00003042.jpg" #"test2/00000000/00000830.jpg"
    total = image_path + path
    input_image = pil.open(total).convert('RGB')
    original_width, original_height = input_image.size
    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    return input_image, (original_height, original_width)

def depthPredict(input_image, origshape, normalize=False, gpu=False):
    with torch.no_grad():
        features = encoder(input_image)
        outputs = depth_decoder(features)
    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(disp, origshape, mode="bilinear", align_corners=False)
    if gpu:
        data = disp_resized.detach().cpu().numpy().squeeze()
    else:
        data = disp_resized.detach().numpy().squeeze()
    if normalize:
        data = (data-data.min())/(data.max()-data.min())
    return data

def prepReadInput(query, bounders=None):
    img = cv2.imread(query)
    ogimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if bounders != None:
        img = getCrop(ogimg, bounders)
    return ogimg.shape, preprocessing(img)

def getCrop(image, bounders):
    xmin, ymin, xmax, ymax = bounders
    img = image[int(ymin):int(ymax)+1,int(xmin):int(xmax)+1] #429.0	2.0	509.0	80.0
    return img

def spherical2cartesial(x):
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])
    return output
preprocessing = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_continuous_gaze_cone3d(head_center_point, gaze_vector, depth_maps, out_sizes):
    """
    head_center_point: tensor(1, 2)
    gaze_vector: tensor(1, 3)
    depth_maps: tensor(1, H, W)
    out_sizes: list(1, 2)
    """
    height, width = out_sizes
    #do i really need to batchify all of this?
    #just assume batch_size=B=1
    B = len(gaze_vector)

    eye_coords = ( #eye_coords must have an additional dimension which represents depth, so might need to cast to int
        (
            head_center_point
            * torch.tensor([out_sizes])
        )
        .int()
    )

    print(f"eye_coords int, {eye_coords}, shape: {eye_coords.shape}")
    head_center_point_depth = depth_maps[0, eye_coords[0,0], eye_coords[0,1]]
    print(f"depth of eye_coords, {head_center_point_depth}")
    eye_coords = torch.cat((eye_coords.float()/torch.tensor([out_sizes]), head_center_point_depth.unsqueeze(-1).unsqueeze(-1)),dim=-1)
    print(f"eye_coords float, {eye_coords}, shape: {eye_coords.shape}")
    gaze_coords = ( #defined by gaze vector which is provided by gaze direction predictor
        (
            #change:(head_center_point + gaze_vector)
            #* torch.tensor([height, width], device=head_center_point.device)
            
            #to:
            (gaze_vector)
        )
        .unsqueeze(1)
        .unsqueeze(1)
    )
    gaze_coords[0,0,0,[0,1]] = -gaze_coords[0,0,0,[1,0]] #swap x and y (rows=y, columns=x) and negate their axes
    print(f"gaze_coords, {gaze_coords}, shape: {gaze_coords.shape}")
    pixel_mat = (
        torch.stack(
            torch.meshgrid(
                [torch.arange(0, 1, 1/height), torch.arange(0, 1, 1/width)], indexing="ij"
            ),
            dim=-1,
        )
        .unsqueeze(0)
        .repeat(head_center_point.shape[0], 1, 1, 1)
        .to(head_center_point.device)
    )
    pixel_mat = torch.cat((pixel_mat, depth_maps.unsqueeze(-1)), dim=-1)
    print(f"pixel_mat shape: {pixel_mat.shape}")

    #changed gaze_coords - eye_coords to just gaze_coords
    dot_prod = torch.sum((pixel_mat - eye_coords) * (gaze_coords), dim=-1)
    gaze_vector_norm = torch.sqrt(torch.sum((gaze_coords) ** 2, dim=-1))
    pixel_mat_norm = torch.sqrt(torch.sum((pixel_mat - eye_coords) ** 2, dim=-1))
    out = dot_prod / (gaze_vector_norm * pixel_mat_norm)

    out = torch.nan_to_num(out)
    mins = torch.min(out.view(B, -1), dim=1).values
    maxs = torch.max(out.view(B, -1), dim=1).values
    print(maxs,mins)
    scaled = (out - mins) / (maxs-mins)

    return scaled



################## put in function ########################
def getDepthAngleCone(dataset, dfSaveLocation, depthSaveLocation, cone=False, gpu=False):
    addData = {"theta":[],"phi":[],"x":[],"y":[],"z":[], "depthLocationNames": []}
    paths = dataset['image_path']
    for x in tqdm(range(len(paths))):
        SOME_PATH = paths.iloc[x]
        input_image, origshape = depthImageLoader(SOME_PATH)
        input_image = input_image.to(deviceDepth)
        data = depthPredict(input_image, (224,224), gpu=gpu) #[height, width]


        curr = dataset[dataset['image_path']==SOME_PATH]
        curr = curr.iloc[0]
        minx = min(curr['head_bbox_x_min'], curr['head_bbox_x_max'])
        maxx = max(curr['head_bbox_x_min'], curr['head_bbox_x_max'])
        miny = min(curr['head_bbox_y_min'], curr['head_bbox_y_max'])
        maxy = max(curr['head_bbox_y_min'], curr['head_bbox_y_max'])
        minx = max(0, minx)
        miny = max(0, miny)
        orig_shape, orig_img = prepReadInput(image_path+SOME_PATH, bounders=[minx, miny, maxx, maxy])

        img = torch.stack([orig_img]*7)
        with torch.no_grad():
            output_gaze, _ = model(img.view(1,7,3,224,224).to("cuda:1"))
        gaze_coords = spherical2cartesial(output_gaze)
        if gpu:
            theta, phi = output_gaze.cpu().numpy().squeeze().tolist()
            x,y,z = gaze_coords.cpu().numpy().squeeze().tolist()
        else:
            theta, phi = output_gaze.numpy().squeeze().tolist()
            x,y,z = gaze_coords.numpy().squeeze().tolist()
        
        if cone:
            gaze_coords[0,2] = -gaze_coords[0,2]

            eyes = curr.iloc[0][['eye_y','eye_x']].values.tolist()
            cone = get_continuous_gaze_cone3d(torch.tensor([eyes]),
                                    gaze_coords, 
                                    torch.from_numpy(data).unsqueeze(0),  #torch.from_numpy(data)
                                    orig_shape[:2]
                                    )
            cone = cone.detach().numpy().squeeze()

        addData['theta'].append(theta)
        addData['phi'].append(phi)
        addData['x'].append(x)
        addData['y'].append(y)
        addData['z'].append(z)
        depthPath = depthSaveLocation+SOME_PATH.replace("/", "-").split(".")[0]+".npy"
        addData['depthLocationNames'] = depthPath
        np.save(depthPath, data)
    dataset['theta'] = addData['theta']
    dataset['phi'] = addData['phi']
    dataset['x'] = addData['x']
    dataset['y'] = addData['y']
    dataset['z'] = addData['z']
    dataset.to_csv(dfSaveLocation)
    
############################################################
getDepthAngleCone(test, 
                  "/nethome/abati7/flash/Data/gazefollow_extended/test2d+a.csv",
                  "/nethome/abati7/flash/Data/gazefollow_extended/test2depth/",
                  cone=False,
                  gpu=True)
