import os
import cv2
import toml
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import utils
from   utils import CONFIG
import networks
from networks.saliency_sampler import Saliency_Sampler
import cv2


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

            
def single_inference(model, modelv, image_dict, return_offset=True):

    with torch.no_grad():
        img_org, trimap = image_dict['image'], image_dict['trimap']
        weight = utils.get_unknown_tensor(trimap)

        alpha_shape = image_dict['alpha_shape']
        
        x, y = trimap.shape[2:]
        idx = torch.argmax(trimap, dim=1)
        idx[idx == 1] = 128
        idx[idx == 0] = 255
        idx[idx == 2] = 0
        trimap_org = idx.view(-1,1, x,y)
        
        tri = trimap_org / 255.
        tri[tri > 0] = 1
        tri_rev = trimap_org / 255.
        tri_rev[tri_rev > 0.7] = 2
        tri_rev[tri_rev < 1] = 1
        tri_rev[tri_rev == 2] = 0
        img_org, trimap_org, tri, tri_rev = img_org.type(torch.FloatTensor).cuda(), trimap_org.type(torch.FloatTensor).cuda(), tri.type(torch.FloatTensor).cuda(), tri_rev.type(torch.FloatTensor).cuda()
        
        trans1, trans2, mask1, mask2 = modelv(torch.cat((img_org, trimap_org/255.), 1),tri,tri_rev)
        trans1 = trans1[:,:3,:,:]
        trans2 = trans2[:,:3,:,:]
        
        image = img_org.cuda()
        trimap = trimap.cuda()
        a,b = image.shape[2:]
        trans1 = nn.Upsample(size=(a,b), mode='bilinear')(trans1)
        trans2 = nn.Upsample(size=(a,b), mode='bilinear')(trans2)
        mask1 = nn.Upsample(size=(a,b), mode='bilinear')(mask1)
        mask2 = nn.Upsample(size=(a,b), mode='bilinear')(mask2)
        
        alpha_pred, bg_pred, fg_pred, info_dict = model(image, trimap, trans1, mask1, trans2, mask2)
        image_origin = image_dict['image_origin'].cuda().unsqueeze(0)

        fg_PRED = fg_pred
        bg_PRED = bg_pred


        fg_PRED[fg_PRED > 1] = 1
        fg_PRED[fg_PRED < 0] = 0
        bg_PRED[bg_PRED > 1] = 1
        bg_PRED[bg_PRED < 0] = 0
        

        if CONFIG.model.trimap_channel == 3:
            trimap_argmax = trimap.argmax(dim=1, keepdim=True)

        alpha_pred[trimap_argmax == 2] = 1
        alpha_pred[trimap_argmax == 0] = 0

        h, w = alpha_shape

        tmp = (image_origin.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        tmp = tmp[32:h+32, 32:w+32, :]

        test_pred = alpha_pred[0, 0, ...].data.cpu().numpy() * 255
        test_pred = test_pred.astype(np.uint8)
        test_pred = test_pred[32:h+32, 32:w+32]

        test_bg = (bg_PRED)*255
        test_bg = test_bg.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)
        test_bg = test_bg[32:h+32, 32:w+32,:]
        
        
        test_fg = (fg_PRED)*255
        test_fg = test_fg.squeeze(0).permute(1,2,0).cpu()#.numpy().astype(np.uint8)
        test_fg = test_fg[32:h+32, 32:w+32,:]

        return test_pred, test_bg, test_fg


def generator_tensor_dict(image_path, trimap_path):
    # read images
    image = cv2.imread(image_path)
    trimap = cv2.imread(trimap_path, 0)
    sample = {'image': image, 'trimap': trimap, 'alpha_shape': trimap.shape, 'image_origin':image}

    # reshape
    h, w = sample["alpha_shape"]
    
    if h % 32 == 0 and w % 32 == 0:
        padded_image = np.pad(sample['image'], ((32,32), (32, 32), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((32,32), (32, 32)), mode="reflect")
        sample['image'] = padded_image
        sample['trimap'] = padded_trimap
    else:
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w
        padded_image = np.pad(sample['image'], ((32,pad_h+32), (32, pad_w+32), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((32,pad_h+32), (32, pad_w+32)), mode="reflect")
        sample['image'] = padded_image
        sample['trimap'] = padded_trimap

    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    # convert GBR images to RGB
    image, trimap = sample['image'][:,:,::-1], sample['trimap']
    # swap color axis
    image = image.transpose((2, 0, 1)).astype(np.float32)
    
    trimap[trimap < 85] = 0
    trimap[trimap >= 170] = 2
    trimap[trimap >= 85] = 1
    # normalize image
    image /= 255.

    # to tensor
    sample['image'], sample['trimap'] = torch.from_numpy(image), torch.from_numpy(trimap).to(torch.long)
    sample['image_origin'] = sample['image'].clone()
    
    sample['image'] = sample['image'].sub_(mean).div_(std)

    if CONFIG.model.trimap_channel == 3:
        sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2, 0, 1).float()
    elif CONFIG.model.trimap_channel == 1:
        sample['trimap'] = sample['trimap'][None, ...].float()
    else:
        raise NotImplementedError("CONFIG.model.trimap_channel can only be 3 or 1")

    # add first channel
    sample['image'], sample['trimap'] = sample['image'][None, ...], sample['trimap'][None, ...]

    return sample

if __name__ == '__main__':
    print('Torch Version: ', torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.toml')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',help="path of checkpoint")
    parser.add_argument('--image-dir', type=str, default='../OUMVLP_recrop/RGB/00001')
    parser.add_argument('--trimap-dir', type=str, default='../OUMVLP_recrop/trimap/00001')
    parser.add_argument('--output',type=str, default='./result/')
    # Parse configuration
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))
    #Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")

    #args.output = os.path.join(args.output, CONFIG.version+'_'+args.checkpoint.split('/')[-1])
    utils.make_dir(args.output)

    # build model
    model = networks.get_generator(encoder=CONFIG.model.arch.encoder, encoder2=CONFIG.model.arch.encoder2, decoder=CONFIG.model.arch.decoder)
    model.cuda()
    modelv = Saliency_Sampler().cuda()

    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    # inference
    model = model.eval()
    foldername = os.listdir(args.image_dir)
    subjectname = args.image_dir.split('/')[3]
    
    for t in range(len(foldername)):
        for image_name in os.listdir(os.path.join(args.image_dir, foldername[t])):
            image_path = os.path.join(os.path.join(args.image_dir, foldername[t]),image_name)
            trimap_path = os.path.join(os.path.join(args.trimap_dir, foldername[t]),image_name)

            print('Image: ', image_path, ' Tirmap: ', trimap_path)
            image_dict = generator_tensor_dict(image_path, trimap_path)
            pred, test_bg, test_fg = single_inference(model, modelv, image_dict)
            h, w = pred.shape[0], pred.shape[1]

            img_org = (image_dict['image_origin']*255.).permute(1,2,0)#.numpy().astype(np.uint8)
            img_org = img_org[32:h+32, 32:w+32, :]
            
            pred_tmp = torch.from_numpy(pred)[None, ...].permute(1,2,0)
            pred_tmp = pred_tmp.repeat(1,1,3)

            greybg = torch.ones(test_bg.shape)*128
            
            dir1 = os.path.join(os.path.join(os.path.join(args.output, 'alpha'),subjectname), foldername[t])
            dir2 = os.path.join(os.path.join(os.path.join(args.output, 'bin_extract'),subjectname), foldername[t])
            dir3 = os.path.join(os.path.join(os.path.join(args.output, 'matt_extract'),subjectname),foldername[t])
            dir4 = os.path.join(os.path.join(os.path.join(args.output, 'pred_bg'),subjectname), foldername[t])
            dir5 = os.path.join(os.path.join(os.path.join(args.output, 'pred_fg'),subjectname), foldername[t])

            
            if not os.path.exists(os.path.join(args.output, 'alpha')):
                os.mkdir(os.path.join(args.output, 'alpha'))
            if not os.path.exists(os.path.join(args.output, 'bin_extract')):
                os.mkdir(os.path.join(args.output, 'bin_extract'))
            if not os.path.exists(os.path.join(args.output, 'matt_extract')):
                os.mkdir(os.path.join(args.output, 'matt_extract'))
            if not os.path.exists(os.path.join(args.output, 'pred_bg')):
                os.mkdir(os.path.join(args.output, 'pred_bg'))
            if not os.path.exists(os.path.join(args.output, 'pred_fg')):
                os.mkdir(os.path.join(args.output, 'pred_fg'))
            
            if not os.path.exists(os.path.join(os.path.join(args.output, 'alpha'),subjectname)):
                os.mkdir(os.path.join(os.path.join(args.output, 'alpha'),subjectname))
            if not os.path.exists(os.path.join(os.path.join(args.output, 'bin_extract'),subjectname)):
                os.mkdir(os.path.join(os.path.join(args.output, 'bin_extract'),subjectname))
            if not os.path.exists(os.path.join(os.path.join(args.output, 'matt_extract'),subjectname)):
                os.mkdir(os.path.join(os.path.join(args.output, 'matt_extract'),subjectname))
            if not os.path.exists(os.path.join(os.path.join(args.output, 'pred_bg'),subjectname)):
                os.mkdir(os.path.join(os.path.join(args.output, 'pred_bg'),subjectname))
            if not os.path.exists(os.path.join(os.path.join(args.output, 'pred_fg'),subjectname)):
                os.mkdir(os.path.join(os.path.join(args.output, 'pred_fg'),subjectname))

            if not os.path.exists(dir1):
                os.mkdir(dir1)
            if not os.path.exists(dir2):
                os.mkdir(dir2)
            if not os.path.exists(dir3):
                os.mkdir(dir3)
            if not os.path.exists(dir4):
                os.mkdir(dir4)
            if not os.path.exists(dir5):
                os.mkdir(dir5)
            cv2.imwrite(os.path.join(dir1, image_name.split('.')[0]+'.jpg'), pred)
            cv2.imwrite(os.path.join(dir2, image_name.split('.')[0]+'.jpg'), (img_org*(pred_tmp/255.)+greybg*(1-pred_tmp/255.)).numpy().astype(np.uint8)[:,:,::-1])
            cv2.imwrite(os.path.join(dir3, image_name.split('.')[0]+'.jpg'), (test_fg*(pred_tmp/255)+greybg*(1-pred_tmp/255)).numpy().astype(np.uint8)[:,:,::-1])
            cv2.imwrite(os.path.join(dir4, image_name.split('.')[0]+'.jpg'), test_bg[:,:,::-1])
            cv2.imwrite(os.path.join(dir5, image_name.split('.')[0]+'.jpg'), test_fg.numpy().astype(np.uint8)[:,:,::-1])
        
