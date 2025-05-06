import os
import sys
from os.path import expanduser
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pplt
import numpy as np

from calculate_PSNR_SSIM import calculate_psnr as psnr
from calculate_PSNR_SSIM import calculate_ssim as ssim

import utils_img
import torch
from torchvision.transforms import functional
from augly.image import functional as aug_functional

def list_images(path):
    images = os.listdir(path)
    
    mask = []
    for image in images:
        is_good = True
        
        full_path = os.path.join(path, image)
    
        if os.path.isdir(full_path): is_good = False
        
        try:
            with Image.open(full_path) as img:
                img.verify()
        except (IOError, SyntaxError):
            is_good = False
            
        mask.append(is_good)
        
    return [os.path.join(path, image) for i, image in enumerate(images) if mask[i]]
    

def get_coin_image(image, show=False, box_th=0.98):
    imc = cv2.imread(image, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(imc, cv2.COLOR_BGR2GRAY)

    imb = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    imm = cv2.morphologyEx(imb, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    
    n, iml = cv2.connectedComponents(imm)
    area = np.asarray([np.sum(iml == i) for i in range(n)])
    idx = np.argsort(-area)

    idx_ = idx[1] if (iml[0, 0] == idx[0]) else idx[0]

    imm = (iml == idx_).astype(np.uint8) * 255

    out_border = cv2.morphologyEx((imm).astype(np.uint8) * 255, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    out_border = ((out_border > 0) & (imm == 0)).astype(np.uint8) * 255
    out_border = cv2.morphologyEx((out_border).astype(np.uint8) * 255, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    n, iml = cv2.connectedComponents(out_border)
    area = np.asarray([np.sum(iml == i) for i in range(n)])
    idx = np.argsort(-area)
    out_border = (iml == idx[1]).astype(np.uint8) * 255

    out_border = cv2.morphologyEx(out_border, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, iml = cv2.connectedComponents(255 - out_border)
    mask_base = (iml == iml[0, 0]).astype(np.uint8) * 255

    cHull = cv2.convexHull(np.argwhere(out_border))    
    cHull = np.flip(cHull, axis=-1)
    mask = np.zeros(mask_base.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cHull], 0, 255, thickness=-1)
    mask = 255 - mask
    
    tmp =  (mask == 0) 
    s0 = np.argwhere(np.sum(tmp, axis=0))
    s1 = np.argwhere(np.sum(tmp, axis=1))
    bbox = np.asarray([s0[0].item(), s0[-1].item(), s1[0].item(), s1[-1].item()], dtype=int)

    bbox_orig = bbox[:]

    box = np.zeros(mask.shape, dtype=np.uint8)
    box[bbox[2]:bbox[3], bbox[0]:bbox[1]] = 255
    n_mask = np.sum((mask == 0) & (box == 255))
    n_box = np.sum(box == 255)

    while n_mask / n_box < box_th:
        bbox = bbox + np.asarray([+5, -5, +5, -5], dtype=int)
        box = np.zeros(mask.shape, dtype=np.uint8)
        box[bbox[2]:bbox[3], bbox[0]:bbox[1]] = 255
        n_mask = np.sum((mask == 0) & (box ==255))
        n_box = np.sum(box == 255)

    if show:
        plt.figure()
        fig, ax = plt.subplots()
        plt.imshow(mask/2 + mask_base/2, cmap='Grays')
        rect = pplt.Rectangle((bbox[0], bbox[2]), bbox[1] - bbox[0], bbox[3] - bbox[2], linewidth=1, edgecolor='r', facecolor='none')
        rect_orig = pplt.Rectangle((bbox_orig[0], bbox_orig[2]), bbox_orig[1] - bbox_orig[0], bbox_orig[3] - bbox_orig[2], linewidth=1, edgecolor='g', facecolor='none')

        ax.add_patch(rect_orig)
        ax.add_patch(rect)
        fig.suptitle(os.path.split(image)[1])         
    
    return imc, bbox, mask, mask_base, bbox_orig


from trustmark import TrustMark

class trustmark_watermarking:
    def __init__(self, mode='P'):
        # Available modes: Q=balance, P=high visual quality, C=compact decoder, B=base from paper
        
        self.tm=TrustMark(verbose=False, model_type=mode, encoding_type=TrustMark.Encoding.BCH_5)

        
    def name(self):
        return "trustmark watermarking"
    
    
    def embed(self, tile, **args):
        cv2.imwrite('tmp0.png', tile)
        
        wm = args['wm']
        orig_w = "".join(["1" if wm[i] else "0" for i in range(len(wm))])

        # encoding example
        cover = Image.open('tmp0.png')
        rgb = cover.convert('RGB')
        has_alpha = cover.mode == 'RGBA'
        if (has_alpha): alpha = cover.split()[-1]
        
        capacity = self.tm.schemaCapacity()
        bitstring = orig_w
        encoded = self.tm.encode(rgb, bitstring, MODE='binary')
        
        if (has_alpha):
          encoded.putalpha(alpha)

        outfile = 'tmp1.png'
        encoded.save(outfile, exif=cover.info.get('exif'), icc_profile=cover.info.get('icc_profile'), dpi=cover.info.get('dpi'))

        tile = cv2.imread('tmp1.png', cv2.IMREAD_COLOR)
        
        return tile
    
    
    def extract(self, tile, **args):
        cv2.imwrite('aux0.png', tile)        
      
        stego = Image.open('aux0.png').convert('RGB')
        wm_secret, wm_present, wm_schema = self.tm.decode(stego, MODE='binary')
        
        os.remove('aux0.png')        
        
        if wm_present:
            wm_l = args['wm_l']
            return [True if wm_secret[i] == '1' else False for i in range(wm_l)]            
        else:
            return None
      

from blind_watermark import WaterMark

class blind_watermarking:
    def name(self):
        return "blind watermarking"
    
    
    def embed(self, tile, **args):
        cv2.imwrite('tmp0.png', tile)        
        wm = args['wm']
    
        bwm1 = WaterMark(password_img=1, password_wm=1)
        bwm1.read_img('tmp0.png')
        bwm1.read_wm(wm, mode='bit')
        bwm1.embed('tmp1.png')
    
        tile = cv2.imread('tmp1.png', cv2.IMREAD_COLOR)
        
        return tile
    
    
    def extract(self, tile, **args):
        cv2.imwrite('aux0.png', tile)        
        wm_l = args['wm_l']
      
        bwm1 = WaterMark(password_img=1, password_wm=1)
        return bwm1.extract('aux0.png', wm_shape=wm_l, mode='bit')


conf_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(conf_path, '../ssl_watermarking'))
    
import encode as sslw_encode
import evaluate as sslw_evaluate
import utils as sslw_utils
import utils_img as sslw_utils_img


class ssl_watermarking:
    def name(self):
        return "ssl watermarking"    

    def __init__(self, **args):
        print("todo")
    

def inject_watermark(image_in, image_out='blind_watermark.jpg', wm=[True, False], use_mask=False, tile_size=None, how=None):    
    if use_mask:
        img, bbox, *_  = get_coin_image(image_in)
        orig = img.copy()
        img = img[bbox[2]:bbox[3], bbox[0]:bbox[1]]
    else:
        img = cv2.imread(image_in, cv2.IMREAD_COLOR)
        orig = img.copy()
    
    if tile_size is None:
        tile_size = img.shape[:2]

    for i in range(0, img.shape[0], tile_size[0]):
        for j in range(0, img.shape[1], tile_size[1]):
            ii = i+tile_size[0]
            jj = j+tile_size[1]
            
            if (ii > img.shape[0]) or (jj > img.shape[1]): continue
                
            tile = img[i:ii, j:jj]            
            img[i:ii, j:jj] = how.embed(tile, wm=wm)
    
    if use_mask:
        orig[bbox[2]:bbox[3], bbox[0]:bbox[1]] = img
    else:
        orig = img

    cv2.imwrite(image_out, orig)
    
    os.remove('tmp0.png')
    os.remove('tmp1.png')

    return


def extract_watermark(image, wm_l=2, use_mask=False, tile_size=None, how=None):
    if use_mask:
        img, bbox, *_  = get_coin_image(image)
        img = img[bbox[2]:bbox[3], bbox[0]:bbox[1]]
    else:
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        
        
    if tile_size is None:
        tile_size = img.shape[:2]
        
    h = {}

    if tile_size[0] == 0:
        print("doh!")

    for i in range(0, img.shape[0], tile_size[0]):
        for j in range(0, img.shape[1], tile_size[1]):
            ii = i+tile_size[0]
            jj = j+tile_size[1]
            
            if (ii > img.shape[0]) or (jj > img.shape[1]): continue
                
            tile = img[i:ii, j:jj]

            wm_extract = how.extract(tile, wm_l=wm_l)

            if wm_extract is None: continue

            l = ('').join(['0' if not(i) else '1' for i in wm_extract])

            if l in h: h[l] += 1
            else: h[l] = 1            
            
    if len(h) == 0: return None
            
    kk = list(h.keys())
    v = np.asarray([h[k] for k in kk])
    s = np.argsort(-v)[0]    
    w = kk[s]
        
    return [w[i] == '1' for i in range(len(w))]     


attacks_dict = {
    "none": lambda x : x,
    "rotation": functional.rotate,
    "grayscale": functional.rgb_to_grayscale,
    "contrast": functional.adjust_contrast,
    "brightness": functional.adjust_brightness,
    "hue": functional.adjust_hue,
    "hflip": functional.hflip,
    "vflip": functional.vflip,
    "blur": functional.gaussian_blur,
    "jpeg": aug_functional.encoding_quality,
    "resize": utils_img.resize,
    "center crop": utils_img.center_crop,
    "overlay emoji": aug_functional.overlay_emoji,
    "random_noise": aug_functional.random_noise, 
    "sharpen": aug_functional.sharpen,
    "generic crop": aug_functional.crop,
}

attacks = [{'attack': 'none'}] \
    + [{'attack': 'generic crop', 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2} for x1 in [0.25, 0] for x2 in [0.8, 1] for y1 in [0.3, 0] for y2 in [0.85, 1]] \
    + [{'attack': 'vflip'}] \
    + [{'attack': 'hflip'}] \
    + [{'attack': 'sharpen', 'factor': f} for f in [1.5, 3, 7, 10]] \
    + [{'attack': 'random_noise', 'var': v} for v in [0.01, 0.02]] \
    + [{'attack': 'overlay emoji', 'emoji_size': s, 'y_pos': 0.5} for s in [0.15, 0.25, 0.35]] \
    + [{'attack': 'overlay emoji', 'emoji_path': 'wm_logo.png', 'y_pos': y, 'x_pos': x} for y in [0.5, 0.35, 0.65] for x in [0.4, 0.6]] \
    + [{'attack': 'rotation', 'angle': r, 'fill': [255, 255, 255]} for r in [2, 15, 30, 45]] \
    + [{'attack': 'center crop', 'scale': s} for s in [0.25, 0.35, 0.5, 0.75, 0.8, 1.2, 1.5]] \
    + [{'attack': 'resize', 'scale': s} for s in [0.5, 0.75, 0.9, 1.1, 1.25]] \
    + [{'attack': 'blur', 'kernel_size': h} for h in [5, 11, 15, 21]] \
    + [{'attack': 'jpeg', 'quality': q} for q in [50, 75, 90, 95, 98, 100]] \
    + [{'attack': 'contrast', 'contrast_factor': c} for c in [0.5, 1., 1.5, 2.]] \
    + [{'attack': 'brightness', 'brightness_factor': b} for b in [0.5, 1., 1.5, 2.]] \
    + [{'attack': 'hue', 'hue_factor': h} for h in [-0.5, -0.25, 0.25, 0.5]] \


ipath = 'coins'
opath = 'wm_coins'
os.makedirs(opath, exist_ok=True)

# watermark key
wm = [False, True, False, False, True, True, False, True]
orig_w = "".join(["1" if wm[i] else "0" for i in range(len(wm))])
l = len(wm)

# border to increase the bounding box
b = 25

# tile size
tl = None

# w_method = blind_watermarking()
w_method = trustmark_watermarking()

crop_image = True

qv = {}

images = list_images(ipath)
for image in images:
    omage = os.path.join(opath, os.path.split(image)[-1])

    if crop_image:
        in_image = 'input_image.png'
        img_orig, bbox, *_  = get_coin_image(image, box_th=.0)

        bbox[0] = max(bbox[0] - b, 0)
        bbox[1] = min(bbox[1] + b, img_orig.shape[0])
        bbox[2] = max(bbox[2] - b, 0)
        bbox[3] = min(bbox[3] + b, img_orig.shape[1])
        
        img_orig = img_orig[bbox[2]:bbox[3], bbox[0]:bbox[1]]
        cv2.imwrite(in_image, img_orig)
    else:
        in_image = image

    inject_watermark(in_image, wm=wm, image_out=omage, tile_size=tl, how=w_method)

    im_orig = cv2.imread(in_image).astype(np.single)
    ow_image = cv2.imread(omage).astype(np.single)
    PSNR = psnr(im_orig, ow_image)
    SSIM = ssim(im_orig, ow_image)
    
    k = os.path.splitext(os.path.split(image)[-1])[0]
    qv[k] = {'PSNR': PSNR, 'SSIM': SSIM}
    
    print(f'{w_method.name()}("{os.path.split(image)[-1]}"): message = "{orig_w}"; PSNR = {PSNR:.3f}; SSIM = {SSIM *100:.3f} %')

    ow_image = Image.open(omage)

    qt = {}
    for q, attack in enumerate(attacks):
        t_image = 'mod.png'

        attack = attack.copy()
        attack_name = attack.pop('attack')

        tw_image = attacks_dict[attack_name](ow_image, **attack)
        tw_image.save(t_image)
        im_wmrk = cv2.imread(t_image).astype(np.single)

        extracted_wm = extract_watermark(t_image, wm_l=l, tile_size=tl, how=w_method)
          
        if extracted_wm is None:
            extr_w = None
            v = 0
        else:
            extr_w = "".join(["1" if extracted_wm[i] else "0" for i in range(len(extracted_wm))])
            v = np.sum(np.asarray(wm) == np.asarray(extracted_wm)) / l    
    
        qt[attack_name + " (" + str(q)  + ")"] = {'msg': extr_w, 'correct bits': v}
    
        print(f'{str(q)}.{attack_name}("{k}"): retrieved message = {extr_w}; {"Pass" if (v == 1) else "Failed"}')
                
    qv[k]['validation'] = qt
    
os.remove('input_image.png')
os.remove('mod.png')
