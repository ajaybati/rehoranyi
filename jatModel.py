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

############### preprocessing funcs ###############
class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocessing = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
def prepReadInput(query):
    IMAGE_PATH = query
    img = Image.open(query)
    img = img.convert("RGB")
    return img.size, preprocessing(img)
##################################################
# ################# helper funcs ###################
def get_soft_head_map(bbox, orig_size, new_size):
    # sizes are height, width
    xmin, ymin, xmax, ymax = bbox
    xmin = xmin * orig_size[0]
    xmax = xmax * orig_size[0]
    ymin = ymin * orig_size[1]
    ymax = ymax * orig_size[1]
    full_map = torch.zeros(orig_size)
    full_map[int(ymin):int(ymax), int(xmin):int(xmax)] = 1.
    resized_map = torch.nn.functional.interpolate(full_map.expand(1,1,-1,-1), new_size, mode='bilinear').squeeze()
    return resized_map

def get_heatmap(gazex, gazey, height, width, sigma=3, htype="Gaussian"):
    # Adapted from https://github.com/ejcgt/attention-target-detection/blob/master/utils/imutils.py

    img = torch.zeros(height, width)
    if gazex < 0 or gazey < 0:  # return empty map if out of frame
        return img
    gazex = int(gazex * width)
    gazey = int(gazey * height)

    # Check that any part of the gaussian is in-bounds
    ul = [int(gazex - 3 * sigma), int(gazey - 3 * sigma)]
    br = [int(gazex + 3 * sigma + 1), int(gazey + 3 * sigma + 1)]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if htype == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    elif htype == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma**2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] += g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    img = img / img.max()  # normalize heatmap so it has max value of 1
    return img

def get_multi_annot_heatmap(gazex, gazey, height, width, sigma=3, htype="Gaussian"):
    img = torch.zeros(height, width)
    num_annot = gazex.shape[0]
    for i in range(num_annot):
        img += get_heatmap(gazex[i], gazey[i], height, width, sigma=sigma, htype=htype)
    img /= float(num_annot)
    return img

def get_one_hot_heatmap(gazex, gazey, height, width):
    target_map = torch.zeros((height, width))
    num_annot = gazex.shape[0]
    for i in range(num_annot):
        if gazex[i] >= 0:
            x, y = map(int,[gazex[i]*width, gazey[i]*height])
            x = min(x, width-1)
            y = min(y, height-1)
            target_map[y, x] = 1
    return target_map

def calculate_auc(heatmap_preds, gt_onehot_heatmaps, sizes):
    #sizes = (height, width)
    hm = heatmap_preds
    gt = gt_onehot_heatmaps
    aucs = []
    for i in range(hm.shape[0]):
        hmresized = transforms.functional.resize(hm[i].unsqueeze(0), sizes[i], antialias=True).squeeze(0)
        # print(hmresized.shape, gt[i].shape)
        aucs.append(sklearn.metrics.roc_auc_score(gt[i].flatten(),hmresized.flatten()))
    
    aucs = np.array(aucs)

    return aucs.mean(), aucs

def calculate_min_and_avg_l2(heatmap_preds, gtgazex, gtgazey):
    heatmaps = heatmap_preds
    total_annot = len(gtgazex)

    flat_argmaxes = heatmaps.flatten().argmax(dim=-1)
    pred_y, pred_x = np.unravel_index(flat_argmaxes, (heatmaps.shape[-2], heatmaps.shape[-1]))

    pred_y = torch.tensor(pred_y) / float(heatmaps.shape[-2])
    pred_x = torch.tensor(pred_x) / float(heatmaps.shape[-1])

    gazex = gtgazex
    gazey = gtgazey

    gt_avg_gazex = gazex.sum() / total_annot
    gt_avg_gazey = gazey.sum() / total_annot

    avg_l2 = torch.sqrt((pred_x - gt_avg_gazex)**2 + (pred_y - gt_avg_gazey)**2)
    
    l2 = torch.sqrt((pred_x - gazex)**2 + (pred_y - gazey)**2)
    min_l2 = l2.min(dim=-1).values
    return min_l2.item(), avg_l2.item()

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

    # print(f"eye_coords int, {eye_coords}, shape: {eye_coords.shape}")
    head_center_point_depth = depth_maps[0, eye_coords[0,0], eye_coords[0,1]]
    # print(f"depth of eye_coords, {head_center_point_depth}")
    eye_coords = torch.cat((eye_coords.float()/torch.tensor([out_sizes]), head_center_point_depth.unsqueeze(-1).unsqueeze(-1)),dim=-1)
    # print(f"eye_coords float, {eye_coords}, shape: {eye_coords.shape}")
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
    # print(f"gaze_coords, {gaze_coords}, shape: {gaze_coords.shape}")
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
    # print(f"pixel_mat shape: {pixel_mat.shape}")

    #changed gaze_coords - eye_coords to just gaze_coords
    dot_prod = torch.sum((pixel_mat - eye_coords) * (gaze_coords), dim=-1)
    gaze_vector_norm = torch.sqrt(torch.sum((gaze_coords) ** 2, dim=-1))
    pixel_mat_norm = torch.sqrt(torch.sum((pixel_mat - eye_coords) ** 2, dim=-1))
    out = dot_prod / (gaze_vector_norm * pixel_mat_norm)

    out = torch.nan_to_num(out)
    mins = torch.min(out.view(B, -1), dim=1).values
    maxs = torch.max(out.view(B, -1), dim=1).values
    # print(maxs,mins)
    scaled = (out - mins) / (maxs-mins)

    return scaled
# ##################################################


class GazefollowTrain(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        readPath = data_dir+"traind+aFINAL.csv"
        self.df = pd.read_csv(readPath, sep=",")
        self.df = self.df[self.df['in_or_out'] == 1] #remove out of frame

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        curr = self.df.iloc[idx]

        size, imageInput = prepReadInput(self.data_dir+curr['image_path'])
        width,height = size


        data = np.load(self.data_dir+"traindepth/"+curr["depthLocationNames"])
        depthMap = (data-data.min())/(data.max()-data.min())
        gaze_coords = curr[['x','y','z']].tolist()
        gaze_coords[2] = -gaze_coords[2] #changing z axis to point forward

        eyes = curr[['eye_y','eye_x']].values.tolist()
        fovMap = get_continuous_gaze_cone3d(torch.tensor([eyes]),
                        torch.tensor([gaze_coords]), 
                        torch.from_numpy(depthMap).unsqueeze(0),  #torch.from_numpy(data)
                        (224,224) #orig_shape[:2]
                        )

        gazex, gazey = np.array([curr['gaze_x']]), np.array([curr['gaze_y']])
        gtHeatmap = get_multi_annot_heatmap(gazex, gazey, 64, 64)
        gtOnehotHeatmap = get_one_hot_heatmap(gazex, gazey, height, width)
        depthMap = torch.from_numpy(depthMap)

        return imageInput, depthMap, fovMap, gtHeatmap, gtOnehotHeatmap, (height, width)
    

class GazefollowTest(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        readPath = data_dir+'test2d+aFINAL.csv'
        self.df = pd.read_csv(readPath, sep=",")
        self.df_dropped = self.df.drop_duplicates(subset=['image_path','eye_x','eye_y'])

    def __len__(self):
        return len(self.df_dropped)

    def __getitem__(self, idx):
        curr = self.df_dropped.iloc[idx]
        all = self.df[self.df['image_path'] == curr['image_path']]

        size, imageInput = prepReadInput(self.data_dir+curr['image_path'])
        width,height = size

        data = np.load(self.data_dir+"test2depth/"+curr["depthLocationNames"])
        depthMap = (data-data.min())/(data.max()-data.min()) 
        gaze_coords = curr[['x','y','z']].tolist()
        gaze_coords[2] = -gaze_coords[2] #changing z axis to point forward

        eyes = curr[['eye_y','eye_x']].values.tolist()
        fovMap = get_continuous_gaze_cone3d(torch.tensor([eyes]),
                        torch.tensor([gaze_coords]), 
                        torch.from_numpy(depthMap).unsqueeze(0),  #torch.from_numpy(data)
                        (224,224) #orig_shape[:2]
                        )

        gazex, gazey = all['gaze_x'].to_numpy(), all['gaze_y'].to_numpy()
        gtOnehotHeatmap = get_one_hot_heatmap(gazex, gazey, height, width)
        depthMap = torch.from_numpy(depthMap)

        return imageInput, depthMap, fovMap, gtOnehotHeatmap, (height, width), gazex, gazey

def collate_fn(data):
    imageInputs, depthMaps, fovMaps, gtHeatmaps, gtOnehotHeatmaps, sizes = zip(*data)
    imageInputs = torch.stack(imageInputs)
    depthMaps = torch.stack(depthMaps)
    fovMaps = torch.stack(fovMaps)
    modelInputs = torch.cat((imageInputs, depthMaps.unsqueeze(1), fovMaps), 1) #[B,5,224,224]
    gtHeatmaps = torch.stack(gtHeatmaps)
    # gtOnehotHeatmaps = torch.stack(gtOnehotHeatmaps)

    return modelInputs.type(torch.FloatTensor), gtHeatmaps, gtOnehotHeatmaps, sizes

def collate_fn_test(data): #imageInput, headmap, heatmap, onehotheatmap, gtangle should all be good for test; fix gazex, gazey
    imageInputs, depthMaps, fovMaps, gtOnehotHeatmaps, sizes, gazex, gazey = zip(*data)
    imageInputs = torch.stack(imageInputs)
    depthMaps = torch.stack(depthMaps)
    fovMaps = torch.stack(fovMaps)

    modelInputs = torch.cat((imageInputs, depthMaps.unsqueeze(1), fovMaps), 1) #[B,5,224,224]
    # gtOnehotHeatmaps = torch.stack(gtOnehotHeatmaps)

    return modelInputs.type(torch.FloatTensor), gtOnehotHeatmaps, gazex, gazey, sizes



from resnetish import Bottleneck
class Horanyi(nn.Module):
    def __init__(self, block = Bottleneck, layers_scene = [3, 4, 6, 3, 2], layers_face = [3, 4, 6, 3, 2]):
        super().__init__()
        self.inplanes_scene = 64
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #Resnet
        #later include NL blocks in layer2_scene and layer3_scene
        self.conv1_scene = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_scene = nn.BatchNorm2d(64)
        self.layer1_scene = self._make_layer_scene(block, 64, layers_scene[0])
        self.layer2_scene = self._make_layer_scene(block, 128, layers_scene[1], stride=2)
        self.layer3_scene = self._make_layer_scene(block, 256, layers_scene[2], stride=2)
        self.layer4_scene = self._make_layer_scene(block, 512, layers_scene[3], stride=2)
        self.layer5_scene = self._make_layer_scene(block, 256, layers_scene[4], stride=1) # additional to resnet50

        # encoding for saliency
        self.compress_conv1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False) #used to be 2048, 1024
        self.compress_bn1 = nn.BatchNorm2d(512) #1024
        self.compress_conv2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False) #used to be 1024, 512
        self.compress_bn2 = nn.BatchNorm2d(512)

        # decoding
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)


    def _make_layer_scene(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_scene != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_scene, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_scene, planes, stride, downsample))
        self.inplanes_scene = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_scene, planes))

        return nn.Sequential(*layers)
        
    def forward(self, imgInputs):
        """
        Input:
            img: concatenated features [r,g,b,d,fov] - tensor[B, 5, 224, 224]

        Returns: 
            outputAngles: (B, 1)
        """
        im = self.conv1_scene(imgInputs)
        im = self.bn1_scene(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.layer1_scene(im)
        im = self.layer2_scene(im)
        im = self.layer3_scene(im)
        im = self.layer4_scene(im)
        scene_feat = self.layer5_scene(im)


        encoding = self.compress_conv1(scene_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)


        x = self.deconv1(encoding)
        x = self.deconv_bn1(x)
        x = self.relu(x)
        
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)

        x = self.conv4(x)

        return x
    

def train_model(num_epochs, train_loader, test_loader, save_dir, devices, output_dir, test=True, save=True, start_epoch_number=0, loadPath=None):
    writer = SummaryWriter(output_dir)

    model = Horanyi() 

    lr = 0.0001

    optimizer_gaze = torch.optim.Adam(model.parameters(),lr=lr)
    if loadPath is not None:
        model.load_state_dict(torch.load(loadPath, map_location="cpu"))
    model = nn.DataParallel(model, device_ids = devices)
    
    model.cuda()

    angle_or_heatmap_loss = nn.MSELoss().cuda()    

    writeNum = len(train_loader) // 150
    
    steps = start_epoch_number*len(train_loader)
    for epoch in range(start_epoch_number, start_epoch_number+num_epochs):

        sum_loss = iter_gaze = aucs = hm_loss = 0

        for i, data in enumerate(tqdm(train_loader)):
            modelInputs, gtHeatmaps, gtOneHotHeatmaps, sizes = data
            modelInputs = Variable(modelInputs).cuda()
            gtHeatmaps = Variable(gtHeatmaps).cuda()

            heatmap_preds = model(modelInputs).squeeze() #[B,1,64,64]
            #deal with loss of heatmaps and angle
            loss = angle_or_heatmap_loss(heatmap_preds, gtHeatmaps)

            sum_loss += loss

            optimizer_gaze.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_gaze.step()
            try:
                auc, all_auc = calculate_auc(heatmap_preds.detach().cpu(), gtOneHotHeatmaps, sizes) #Question: how and when do I record this in tb?
            except Exception as e:
                print(str(e))
                print(heatmap_preds)
                sys.exit()
            aucs += auc

            iter_gaze += 1
            steps += 1

            if (i+1) % writeNum == 0: #step-level train loss THIS IS WHERE TENSORBOARD SHOULD BE WRITTEN TO (EVERY 500 DATAPOINTS?)
                writer.add_scalar("Step Loss Total-Train", sum_loss/iter_gaze, steps)
                writer.add_scalar("Step Loss HM-Train", hm_loss/iter_gaze, steps)
                writer.add_scalar("Step AUC-Train", aucs/iter_gaze, steps)

        if save:
            if epoch % 1 == 0: #epoch-level train loss logging on tensorboard
                print('Taking snapshot...',
                    torch.save(model.module.state_dict(),f"{save_dir}/_epoch_{epoch+1}_loss_{sum_loss/iter_gaze:.2f}.pkl"))
                writer.add_scalar("Epoch Loss-Train", sum_loss/iter_gaze, epoch+1)
                writer.add_scalar("Epoch Loss HM-Train", hm_loss/iter_gaze, epoch+1)
                writer.add_scalar("Epoch AUC-Train", aucs/iter_gaze, epoch+1)
        if test:
            #epoch-level test loss logging on tensorboard
            (min_l2, avg_l2), avg_auc = test_model(model, test_loader)
            writer.add_scalar("Epoch Min L2-Test", min_l2, epoch+1)
            writer.add_scalar("Epoch Avg L2-Test", avg_l2, epoch+1)
            writer.add_scalar("Epoch AUC-Test", avg_auc, epoch+1)


def test_model(model, test_dataloader):
    model.eval()
    min_l2 = 0
    avg_l2 = 0
    auc_mean = 0
    for i, data in enumerate(tqdm(test_dataloader)):
        modelInputs, gtOnehotHeatmaps, gazex, gazey, sizes = data
        modelInputs = Variable(modelInputs).cuda()

        heatmap_preds = model(modelInputs).squeeze() #[B,1,64,64]


        #auc calculation
        auc, all_auc = calculate_auc(heatmap_preds.detach().cpu(), gtOnehotHeatmaps, sizes)
        auc_mean += np.sum(all_auc)

        #l2 calculation
        for i in range(len(gazex)):
            ml2, al2 = calculate_min_and_avg_l2(heatmap_preds[i].detach().cpu(),gazex[i],gazey[i])
            min_l2 += ml2
            avg_l2 += al2
    model.train()
    return (min_l2/len(gtest),avg_l2/len(gtest)), auc_mean/len(gtest)

batch_size = 128
devices = [0,1]

data_path = "/nethome/abati7/flash/Data/gazefollow_extended/"
gtrain = GazefollowTrain(data_path)
gtest = GazefollowTest(data_path)
train_dataloader = DataLoader(gtrain, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
test_dataloader = DataLoader(gtest, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_test, num_workers=2)

train_model(10,
            train_dataloader,
            test_dataloader,
            '/nethome/abati7/flash/Work/jat/experiments/fullnoNL2',
            devices,
            '/nethome/abati7/flash/Work/jat/experimentsTB/fullnoNL2', 
            test=True, save=True, start_epoch_number=0, loadPath=None)

