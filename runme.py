import os
import sys
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pplt
import numpy as np
import wget
import gdown
import zipfile
import argparse
from onedrivedownloader import download as onedrive_download
from skimage.transform import resize as skimage_resize
import pickle
from datetime import datetime

import bchlib
import tensorflow as tf
import tensorflow.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import torchvision.transforms.functional as TF

from calculate_PSNR_SSIM import calculate_psnr as psnr
from calculate_PSNR_SSIM import calculate_ssim as ssim

import utils_img
import torch
from torchvision.transforms import functional
from torchvision.transforms import ToPILImage
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
    def __init__(self, **args):
        # Available modes: Q=balance, P=high visual quality, C=compact decoder, B=base from paper
        
        mode='P'
        if 'mode' in args: mode = args['mode']
        
        self.tm=TrustMark(verbose=False, model_type=mode, encoding_type=TrustMark.Encoding.BCH_5)

        self.wm_l = 30        
        if "wm_l" in args: self.wm_l = args["wm_l"]

        
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
        
        # capacity = self.tm.schemaCapacity()
        bitstring = orig_w
        encoded = self.tm.encode(rgb, bitstring, MODE='binary')
        
        if (has_alpha):
          encoded.putalpha(alpha)

        outfile = 'tmp1.png'
        encoded.save(outfile, exif=cover.info.get('exif'), icc_profile=cover.info.get('icc_profile'), dpi=cover.info.get('dpi'))

        tile = cv2.imread('tmp1.png', cv2.IMREAD_COLOR)
        
        os.remove('tmp0.png')          
        os.remove('tmp1.png')            
        
        return tile
    
    
    def extract(self, tile, **args):
        cv2.imwrite('aux0.png', tile)        
      
        stego = Image.open('aux0.png').convert('RGB')
        wm_secret, wm_present, wm_schema = self.tm.decode(stego, MODE='binary')
        
        os.remove('aux0.png')        
        
        if wm_present:
            wm_l = args['wm_l']
            if wm_l is None: wm_l = self.wm_l
            
            return [True if wm_secret[i] == '1' else False for i in range(wm_l)]            
        else:
            return None
      

from blind_watermark import WaterMark, bw_notes
bw_notes.close()

class blind_watermarking:
    def name(self):
        return "blind watermarking"
    
    def __init__(self, **args):    
        self.wm_l = 30
        
        if "wm_l" in args: self.wm_l = args["wm_l"]

            
    def embed(self, tile, **args):
        cv2.imwrite('tmp0.png', tile)        
        wm = args['wm']
    
        bwm1 = WaterMark(password_img=1, password_wm=1)
        bwm1.read_img('tmp0.png')
        bwm1.read_wm(wm, mode='bit')
        bwm1.embed('tmp1.png')
    
        tile = cv2.imread('tmp1.png', cv2.IMREAD_COLOR)
        
        os.remove('tmp0.png')          
        os.remove('tmp1.png')          
        
        return tile
    
    
    def extract(self, tile, **args):
        cv2.imwrite('aux0.png', tile)        

        wm_l = None
        if "wm_l" in args: wm_l = args["wm_l"]
        if wm_l is None: wm_l = self.wm_l
        
        bwm1 = WaterMark(password_img=1, password_wm=1)
        wtm = bwm1.extract('aux0.png', wm_shape=wm_l, mode='bit')

        os.remove('aux0.png')

        return wtm


conf_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(conf_path, 'ssl_watermarking'))
    
import encode as sslw_encode
import decode as sslw_decode
import data_augmentation as sslw_data_augmentation
import utils as sslw_utils
import utils_img as sslw_utils_img

class ssl_watermarking:
    def name(self):
        return "ssl watermarking"    

    def __init__(self, **args):
        # Set seeds for reproductibility
        torch.manual_seed(0)
        np.random.seed(0)
        
        os.makedirs("sslwm/models", exist_ok=True)
        os.makedirs("sslwm/normlayers", exist_ok=True)
        
        to_check = [
            [
                "sslwm/models/dino_r50_plus.pth",
                "https://dl.fbaipublicfiles.com/ssl_watermarking/dino_r50_plus.pth",
            ],
            [            
                "sslwm/normlayers/out2048_yfcc_orig.pth",
                "https://dl.fbaipublicfiles.com/ssl_watermarking/out2048_yfcc_orig.pth",
            ],
            [
                "sslwm/normlayers/out2048_yfcc_resized.pth",
                "https://dl.fbaipublicfiles.com/ssl_watermarking/out2048_yfcc_resized.pth",
            ],
            [
                "sslwm/normlayers/out2048_coco_orig.pth",
                "https://dl.fbaipublicfiles.com/ssl_watermarking/out2048_coco_orig.pth",
            ],
            [
                "sslwm/normlayers/out2048_coco_resized.pth",
                "https://dl.fbaipublicfiles.com/ssl_watermarking/out2048_coco_resized.pth",
            ],
        ]
        
        for file, link in to_check:
            if not os.path.isfile(file):
                wget.download(link, out=file)
            
        self.model_path = "sslwm/models/dino_r50_plus.pth"
        self.model_name = 'resnet50'
        self.normlayer_path = "sslwm/normlayers/out2048_yfcc_orig.pth"
        self.carrier_dir = "sslwm/carriers"
        self.device = "cuda"
        self.num_bits = 32
        self.data_augmentation = True
        
        parser = argparse.ArgumentParser()
        
        def aa(*args, **kwargs):
            group.add_argument(*args, **kwargs)        

        group = parser.add_argument_group('Experiments parameters')        
        aa("--verbose", type=int, default=0)
        
        group = parser.add_argument_group('Marking parameters')
        aa("--target_psnr", type=float, default=42.0, help="Target PSNR value in dB. (Default: 42 dB)")
        aa("--target_fpr", type=float, default=1e-6, help="Target FPR of the dectector. (Default: 1e-6)")
                
        group = parser.add_argument_group('Optimization parameters')
        aa("--epochs", type=int, default=100, help="Number of epochs for image optimization. (Default: 100)")
        aa("--data_augmentation", type=str, default="all", choices=["none", "all"], help="Type of data augmentation to use at marking time. (Default: All)")
        aa("--optimizer", type=str, default="Adam,lr=0.01", help="Optimizer to use. (Default: Adam,lr=0.01)")
        aa("--scheduler", type=str, default=None, help="Scheduler to use. (Default: None)")
        aa("--batch_size", type=int, default=1, help="Batch size for marking. (Default: 128)")
        aa("--lambda_w", type=float, default=5e4, help="Weight of the watermark loss. (Default: 1.0)")
        aa("--lambda_i", type=float, default=1.0, help="Weight of the image loss. (Default: 1.0)")
        
        self.params = parser.parse_args()
        
        if "model_path" in args: self.model_path = args["model_path"]
        if "model_name" in args: self.model_name = args["model_name"]
        if "normlayer_path" in args: self.model_path = args["normlayer_path"]
        if "carrier_dir" in args: self.carrier_dir = args["carrier_dir"]
        if "device" in args: self.device = args["device"]
        if "wm_l" in args: self.num_bits = args["wm_l"]

        if self.device != 'cpu': 
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Loads backbone and normalization layer
        backbone = sslw_utils.build_backbone(path=self.model_path, name=self.model_name)
        normlayer = sslw_utils.load_normalization_layer(path=self.normlayer_path)
        self.model = sslw_utils.NormLayerWrapper(backbone, normlayer)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        
        # Load or generate carrier and angle
        if not os.path.exists(self.carrier_dir):
            os.makedirs(self.carrier_dir, exist_ok=True)
        D = self.model(torch.zeros((1, 3, 224, 224)).to(self.device)).size(-1)
        K = self.num_bits
        carrier_path = os.path.join(self.carrier_dir,'carrier_%i_%i.pth'%(K, D))        
        if os.path.exists(carrier_path):
            self.carrier = torch.load(carrier_path)
            assert D == self.carrier.shape[1]
        else:
            self.carrier = sslw_utils.generate_carriers(K, D, output_fpath=carrier_path)
        self.carrier = self.carrier.to(self.device, non_blocking=True) # direction vectors of the hyperspace

        if self.data_augmentation: self.data_aug = sslw_data_augmentation.All()
        else: self.data_aug = sslw_data_augmentation.DifferentiableDataAugmentation()


    def embed(self, tile, **args):
        os.makedirs('sslw_tmp/sslw', exist_ok=True)
        
        cv2.imwrite('sslw_tmp/sslw/tmp0.png', tile)        
        wm = args['wm']
        
        msgs = torch.tensor([wm])

        # encoding example

        dataloader = sslw_utils_img.get_dataloader('sslw_tmp', batch_size=self.params.batch_size)

        pt_img = sslw_encode.watermark_multibit(dataloader, msgs, self.carrier, self.model, self.data_aug, self.params)
        imgs_out = ToPILImage()(sslw_utils_img.unnormalize_img(pt_img[0]).squeeze(0)) 
        imgs_out.save('tmp1.png')
        
        tile = cv2.imread('tmp1.png', cv2.IMREAD_COLOR)
        
        os.remove('sslw_tmp/sslw/tmp0.png')          
        os.rmdir('sslw_tmp/sslw')          
        os.rmdir('sslw_tmp')          
        os.remove('tmp1.png')          

        return tile

        
    def extract(self, tile, **args):   
        cv2.imwrite('aux0.png', tile)
        img = Image.open('aux0.png')
        
        decoded_data = sslw_decode.decode_multibit([img], self.carrier, self.model)
        
        os.remove('aux0.png')
                
        return decoded_data[0]['msg']

        
from imwatermark import WatermarkEncoder
from imwatermark import WatermarkDecoder

class invisible_watermarking:
    def name(self):
        return self.method + " watermarking"
    
    def __init__(self, **args):    
        self.wm_l = 30        
        if "wm_l" in args: self.wm_l = args["wm_l"]
        
        self.method = 'rivaGan' # dwtDct|dwtDctSvd|rivaGan
        if "method" in args: self.method = args['method']
        
        if self.method == 'rivaGan':
            WatermarkEncoder.loadModel()
            WatermarkDecoder.loadModel()
            
            
    def embed(self, tile, **args):
        cv2.imwrite('tmp0.png', tile)        
        wm = args['wm']

        bgr = cv2.imread('tmp0.png', cv2.IMREAD_COLOR)
        
        if self.method == 'rivaGan':
            wm_ = [False] * 32
            for i, v in enumerate(wm):
                wm_[i] = v
            wm = wm_

        wm = [1 if v else 0 for v in wm]

        encoder = WatermarkEncoder()        
        encoder.set_watermark('bits', wm)
        bgr_encoded = encoder.encode(bgr, self.method)
        cv2.imwrite('tmp1.png', bgr_encoded)
    
        tile = cv2.imread('tmp1.png', cv2.IMREAD_COLOR)
        
        os.remove('tmp0.png')          
        os.remove('tmp1.png')          
        
        return tile
    
    
    def extract(self, tile, **args):
        cv2.imwrite('aux0.png', tile)        

        wm_l = None
        if "wm_l" in args: wm_l = args["wm_l"]
        if wm_l is None: wm_l = self.wm_l  
        
        wm_l_ = wm_l
        if self.method == 'rivaGan': wm_l_ = 32
        
        bgr = cv2.imread('aux0.png', cv2.IMREAD_COLOR)
        
        decoder = WatermarkDecoder('bits', wm_l_)
        watermark = decoder.decode(bgr, self.method)
        wtm = [True if v else False for v in watermark]
        wtm = wtm[:wm_l]
                
        os.remove('aux0.png')

        return wtm


conf_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(conf_path, 'stegastamp'))
    
class stegastamp_watermarking:
    def name(self):
        return "stegastamp watermarking"    

    def __init__(self, **args):
        # Set seeds for reproductibility
        torch.manual_seed(0)
        np.random.seed(0)
        
        self.off = 5
        if "off" in args: self.off = args["off"]
                
        self.wm_l = 30        
        if "wm_l" in args: self.wm_l = args["wm_l"]               
        
        os.makedirs("stegam", exist_ok=True)
        
        to_check = [
            [
                "stegam/saved_models",
                "https://unipa-my.sharepoint.com/:u:/g/personal/fabio_bellavia_unipa_it/ESjEjrcKYldLu-uyAnuAmLgB9Ckn-5utd9QcX4GMerdGcQ?e=xe58p8"
            ],
        ]
        
        for path, link in to_check:
            if not os.path.isdir(path):
                onedrive_download(link, filename="stegam/stegastamp_model.zip", unzip=True, unzip_path='stegam')
                
        model ="stegam/saved_models/stegastamp_pretrained"
        self.sess = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        self.model = tf.compat.v1.saved_model.loader.load(self.sess, [tag_constants.SERVING], model)

        input_secret_name = self.model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
        input_image_name = self.model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
        self.input_secret = tf.compat.v1.get_default_graph().get_tensor_by_name(input_secret_name)
        self.input_image = tf.compat.v1.get_default_graph().get_tensor_by_name(input_image_name)
    
        output_stegastamp_name = self.model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
        output_residual_name = self.model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
        self.output_stegastamp = tf.compat.v1.get_default_graph().get_tensor_by_name(output_stegastamp_name)
        self.output_residual = tf.compat.v1.get_default_graph().get_tensor_by_name(output_residual_name)

        output_secret_name = self.model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
        self.output_secret = tf.compat.v1.get_default_graph().get_tensor_by_name(output_secret_name)
    
        self.width = 400
        self.height = 400

        self.BCH_POLYNOMIAL = 137
        self.BCH_BITS = 5
    
        self.bch = bchlib.BCH(self.BCH_BITS, prim_poly=self.BCH_POLYNOMIAL)
                
                
    def embed(self, tile, **args):
        wm = args['wm']

        wm_ = [False] * 56
        for i, v in enumerate(wm):
            wm_[i] = v
        wm = wm_
        wm = [1 if v else 0 for v in wm]

        wsecret = np.packbits(np.asarray(wm)).tobytes()
        
        data = bytearray(wsecret)
        ecc = self.bch.encode(data)
        packet = data + ecc
    
        packet_binary = ''.join(format(x, '08b') for x in packet)
        secret = [int(x) for x in packet_binary]
        secret.extend([0,0,0,0])
    
        cv2.imwrite('tmp0.png', tile)       
        # size = (self.width, self.height)
        
        image = Image.open('tmp0.png').convert("RGB")  
        orig_image = np.array(image, copy=True, dtype=np.float32) / 255.
                
        image = np.array(image, dtype=np.float32) / 255.
        # image = skimage_resize(image, (self.height, self.width), anti_aliasing=True)

        offy = (image.shape[0] - self.height) // 2
        offx = (image.shape[1] - self.width) // 2
        image = image[offy:offy+self.height, offx:offx+self.width]
    
        feed_dict = {self.input_secret:[secret],
                     self.input_image:[image]}
    
        hidden_img, residual = self.sess.run([self.output_stegastamp, self.output_residual], feed_dict=feed_dict)
    
        # rescaled = (hidden_img[0] * 255).astype(np.uint8)
        # residual = residual[0]+.5
        # residual = (residual * 255).astype(np.uint8)

        # diff = hidden_img[0] - image
        # diff_resized = skimage_resize(diff, orig_image.shape[:2], anti_aliasing=True)
        # final_image = ((orig_image + diff_resized) * 255).astype(np.uint8)         

        # rescaled = hidden_img[0]
        # rescaled_resized = skimage_resize(rescaled, orig_image.shape[:2], anti_aliasing=True)
        # final_image = (rescaled_resized * 255).astype(np.uint8)         

        o = self.off
        orig_image[offy+o:offy+self.height-o, offx+o:offx+self.width-o] = hidden_img[0][o:-o,o:-o]
        final_image = (orig_image * 255).astype(np.uint8)      

        im = Image.fromarray(np.array(final_image))
        im.save('tmp1.png')

        tile = cv2.imread('tmp1.png', cv2.IMREAD_COLOR)
        
        os.remove('tmp0.png')          
        os.remove('tmp1.png')          

        return tile

        
    def extract(self, tile, **args):    
        offy = (tile.shape[0] - self.height) // 2
        offx = (tile.shape[1] - self.width) // 2
        
        pady = 0 if (offy >= 0) else -offy
        padx = 0 if (offx >= 0) else -offx

        tile = np.pad(tile, ((pady, pady), (padx, padx), (0, 0)), 'constant', constant_values=0)
        
        offy = max(0, offy)
        offx = max(0, offx)
                
        tile_ = tile[offy:offy+self.height, offx:offx+self.width]

        # tile_ = skimage_resize(tile.astype(np.float32) / 255., (self.height, self.width), anti_aliasing=True)
        
        cv2.imwrite('aux0.png', tile_.astype(np.uint8))
        image = np.array(Image.open('aux0.png').convert("RGB"), dtype=np.float32) / 255
        
        feed_dict = {self.input_image:[image]}

        secret = self.sess.run([self.output_secret], feed_dict=feed_dict)[0][0]

        packet_binary = "".join([str(int(bit)) for bit in secret[:96]])
        # packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
        # packet = bytearray(packet)

        # data, ecc = packet[:-self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]
        # bitflips = self.bch.decode(data, ecc)
        # if bitflips != -1: print("errors in message!")

        wm_l = None
        if "wm_l" in args: wm_l = args["wm_l"]
        if wm_l is None: wm_l = self.wm_l  

        z = packet_binary[:wm_l]
        msg = [True if v == '1' else False for v in z]

        os.remove('aux0.png')
                
        return msg


conf_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(conf_path, 'arwgan'))
    
import arwgan.utils as arwgan_utils
import model.ARWGAN as arwgan

class arwgan_watermarking:
    def name(self):
        return "arwgan watermarking"    

    def __init__(self, **args):        
        self.wm_l = 30   
        if "wm_l" in args: self.wm_l = args["wm_l"]               

        self.device = 'cuda'
        if "device" in args: self.device = args["device"]

        if self.device != 'cpu': 
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs("arwganm", exist_ok=True)
        
        to_check = [
            [
                "arwganm/pretrain",
                "https://drive.google.com/file/d/1jDpF0LBmuFiy4PNvqaaz7vXyHCbHA4ao/view?usp=drive_link"
            ],
        ]
        
        for path, link in to_check:
            if not os.path.isdir(path):
                gdown.download(link, "arwganm/pretrain.zip", fuzzy=True)
                
            with zipfile.ZipFile("arwganm/pretrain.zip", "r") as zip_ref:
                zip_ref.extractall("arwganm")

        with torch.no_grad():
            _, self.net_config, noise_config = arwgan_utils.load_options('arwganm/pretrain/options-and-config.pickle')
            noise_config = []
            noiser = arwgan.Noiser(noise_config, self.device)
    
            checkpoint = torch.load("arwganm/pretrain/checkpoints/ARWGAN.pyt", map_location=torch.device('cpu'))
            self.hidden_net = arwgan.ARWGAN(self.net_config, self.device, noiser, None)
            
            arwgan_utils.model_from_checkpoint(self.hidden_net, checkpoint)
            self.hidden_net.encoder_decoder.eval()


    def embed(self, tile, **args):
        cv2.imwrite('tmp0.png', tile)        
        wm = args['wm']

        image_pil = Image.open('tmp0.png')
        image = TF.to_tensor(image_pil).to(self.device)
        image_ = image * 2 - 1
        
        wm_ = [False] * self.net_config.message_length
        for i, v in enumerate(wm):
            wm_[i] = v
        wm = wm_

        wm = [1 if v else 0 for v in wm]

        with torch.no_grad():
            image_ = image_.unsqueeze_(0)
            message = torch.Tensor(wm).unsqueeze(0).to(self.device)
            enc_img = self.hidden_net.encoder_decoder.encoder(image_, message)
           
        image = enc_img.squeeze(0)  
        image = ((image + 1) * 127.5).round().clip(0, 255).permute(1, 2, 0).to('cpu').numpy().astype(np.uint8)

        im = Image.fromarray(np.array(image))
        im.save('tmp1.png')

        tile = cv2.imread('tmp1.png', cv2.IMREAD_COLOR)

        os.remove('tmp0.png')          
        os.remove('tmp1.png')   

        return tile               
        
            
    def extract(self, tile, **args):    
        wm_l = None
        if "wm_l" in args: wm_l = args["wm_l"]
        if wm_l is None: wm_l = self.wm_l  
          
        cv2.imwrite('aux0.png', tile.astype(np.uint8))

        image_pil = Image.open('aux0.png')
        image = TF.to_tensor(image_pil).to(self.device)
        image = (image * 2 - 1).unsqueeze(0)

        with torch.no_grad():
            decoded_messages = self.hidden_net.encoder_decoder.decoder(image).squeeze(0)
            packet_binary = decoded_messages.detach().cpu().numpy().round().clip(0, 1)

        wm_l = None
        if "wm_l" in args: wm_l = args["wm_l"]
        if wm_l is None: wm_l = self.wm_l  

        z = packet_binary[:wm_l].astype(int)
        msg = [True if v == 1 else False for v in z]

        os.remove('aux0.png')
                
        return msg


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

    if not isinstance(how, list): how = [how]

    for i in range(0, img.shape[0], tile_size[0]):
        for j in range(0, img.shape[1], tile_size[1]):
            ii = i+tile_size[0]
            jj = j+tile_size[1]
            
            if (ii > img.shape[0]) or (jj > img.shape[1]): continue
                
            tile = img[i:ii, j:jj]
            for hv in how: tile = hv.embed(tile, wm=wm)                    
            img[i:ii, j:jj] = tile
    
    if use_mask:
        orig[bbox[2]:bbox[3], bbox[0]:bbox[1]] = img
    else:
        orig = img

    cv2.imwrite(image_out, orig)
    
    return


def extract_watermark(image, wm_l=None, use_mask=False, tile_size=None, how=None):
    if use_mask:
        img, bbox, *_  = get_coin_image(image)
        img = img[bbox[2]:bbox[3], bbox[0]:bbox[1]]
    else:
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        
        
    if tile_size is None:
        tile_size = img.shape[:2]

    if not isinstance(how, list): how = [how]
    h = []
    for k in range(len(how)): h.append({})
	
    if tile_size[0] == 0:
        print("doh!")

    for i in range(0, img.shape[0], tile_size[0]):
        for j in range(0, img.shape[1], tile_size[1]):
            ii = i+tile_size[0]
            jj = j+tile_size[1]
            
            if (ii > img.shape[0]) or (jj > img.shape[1]): continue
                
            tile = img[i:ii, j:jj]

            wm_extract = []
            for hv in how: wm_extract.append(hv.extract(tile, wm_l=wm_l))            

            for k in range(len(how)):
                if wm_extract[k] is None: continue

                l = ('').join(['0' if not(i) else '1' for i in wm_extract[k]])

                if l in h[k]: h[k][l] += 1
                else: h[k][l] = 1            

    rval = []

    for k in range(len(how)):
        if len(h[k]) == 0: continue
            
        kk = list(h[k].keys())
        v = np.asarray([h[k][i] for i in kk])
        s = np.argsort(-v)[0]    
        w = kk[s]
        
        rval.append([w[i] == '1' for i in range(len(w))])     

    return rval


def watermark_functional(img_orig, wm=None, w_method=None, tile_size=None):
    if wm is None: wm = [False] * 30

    if w_method is None: return img_orig

    img_orig.save('tmp_in.png')	
    inject_watermark('tmp_in.png', wm=wm, image_out='tmp_out.png', tile_size=tile_size, how=w_method)
    return Image.open('tmp_out.png')


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
    "random noise": aug_functional.random_noise, 
    "sharpen": aug_functional.sharpen,
    "generic crop": aug_functional.crop,
    "aspect ratio": aug_functional.change_aspect_ratio,
    "re-watermark": watermark_functional,
}

# watermark key
# coins = 3 - 15 - 9 - 14 -19
orig_w = "0001101111010010111010011"
wm = [True if v == '1' else False for v in orig_w]
l = len(wm)

# watermarking methods initialization
w_list = [
    ssl_watermarking(wm_l=l),
    blind_watermarking(wm_l=l),
    trustmark_watermarking(wm_l=l),
    invisible_watermarking(method='dwtDct', wm_l=l),
    invisible_watermarking(method='dwtDctSvd', wm_l=l),
    invisible_watermarking(method='rivaGan', wm_l=l),
    stegastamp_watermarking(wm_l=l),
    arwgan_watermarking(wm_l=l),
    ]

# watermarking methods to test
w_methods = []
for w_el1 in w_list:
    for w_el2 in w_list:
        if w_el1.name() == w_el2.name(): continue
        w_methods.append([w_el1, w_el2]) 
w_methods = [w_el for w_el in w_list] + w_methods

# attacks to evaluate
attacks = [{'attack': 'none'}] \
    + [{'attack': 'aspect ratio', 'ratio': r} for r in [0.75, 0.85, 0.95, 1.05, 1.15, 1.25]] \
    + [{'attack': 'resize', 'scale': s} for s in [0.5, 0.75, 0.9, 1.1, 1.25]] \
    + [{'attack': 'generic crop', 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2} for x1 in [0.25, 0] for x2 in [0.8, 1] for y1 in [0.3, 0] for y2 in [0.85, 1]] \
    + [{'attack': 'center crop', 'scale': s} for s in [0.25, 0.35, 0.5, 0.75, 0.8, 1.2, 1.5]] \
    + [{'attack': 'vflip'}] \
    + [{'attack': 'hflip'}] \
    + [{'attack': 'overlay emoji', 'emoji_size': s, 'y_pos': y, 'x_pos': x} for s in [0.15, 0.25, 0.35] for y in [0.5, 0.2] for x in [0.5, 0.7]] \
    + [{'attack': 'overlay emoji', 'emoji_path': 'data/wm_logo.png', 'y_pos': y, 'x_pos': x} for y in [0.5, 0.1, 0.35, 0.65] for x in [0.1, 0.4, 0.6]] \
    + [{'attack': 'rotation', 'angle': r, 'fill': [255, 255, 255]} for r in [2, 15, 30, 45]] \
    + [{'attack': 're-watermark', 'wm': [False] * l, 'w_method': method} for method in w_list] \
    + [{'attack': 'sharpen', 'factor': f} for f in [1.5, 3, 7, 10]] \
    + [{'attack': 'random noise', 'var': v} for v in [0.01, 0.02]] \
    + [{'attack': 'blur', 'kernel_size': h} for h in [5, 11, 15, 21]] \
    + [{'attack': 'jpeg', 'quality': q} for q in [20, 50, 75, 90, 95, 98, 100]] \
    + [{'attack': 'contrast', 'contrast_factor': c} for c in [0.5, 1., 1.5, 2.]] \
    + [{'attack': 'brightness', 'brightness_factor': b} for b in [0.5, 1., 1.5, 2.]] \
    + [{'attack': 'hue', 'hue_factor': h} for h in [-0.5, -0.25, 0.25, 0.5]] \

# coin images input path prefix
ipath = 'coins' # + '_full'

# watermarked coin images output path prefix
opath = 'wm_' + ipath

# border to increase the bounding box when removing the white background 
b = 25

# tile size
tl = None

# remove background
crop_image = True

# save watermarked output as unlossy
unlossy = True

# out file name
out_file = datetime.today().strftime('%Y-%m-%d-%H:%M:%S') + '.pkl'

qv = {}
images = list_images(ipath)
for image in images:
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

    k = os.path.splitext(os.path.split(image)[-1])[0]
    qv[k] = {}        
    for w_method in w_methods:
        if isinstance(w_method, list):
            w = '+'.join([ww_method.name() for ww_method in w_method])
        else:
            w = w_method.name()
        
        opath_ = opath + ' ' + w
        os.makedirs(opath_, exist_ok=True)

        omage = os.path.join(opath_, os.path.split(image)[-1])
        if unlossy: omage = omage[:-4] + '.png'

        inject_watermark(in_image, wm=wm, image_out=omage, tile_size=tl, how=w_method)
    
        im_orig = cv2.imread(in_image).astype(np.single)
        ow_image = cv2.imread(omage).astype(np.single)
        PSNR = psnr(im_orig, ow_image)
        SSIM = ssim(im_orig, ow_image)
        
        qv[k][w] = {'PSNR': PSNR, 'SSIM': SSIM}
        
        print(f'{w} on "{os.path.split(image)[-1]}": message = "{orig_w}"; PSNR = {PSNR:.3f}; SSIM = {SSIM *100:.3f} %')
    
        ow_image = Image.open(omage)
    
        qt = {}
        for q, full_attack in enumerate(attacks):
            t_image = str(q) + 'mod.png'
            tw_image = ow_image
    
            if not isinstance(full_attack, list): full_attack = [full_attack]
            full_attack_name = ''
    
            for q_, attack in enumerate(full_attack):    
                attack = attack.copy()
                attack_name = attack.pop('attack')

                params = []
                for pk in attack.keys():
                    if isinstance(attack[pk], (int,float)): params.append(str(attack[pk]))
                    elif pk == 'w_method': params.append(attack[pk].name())
                params = '(' + ','.join(params)  + ')' if len(params) else ''

                full_attack_name += ('+' if q_ else '') + attack_name + params
                
                tw_image = attacks_dict[attack_name](tw_image, **attack)
                tw_image.save(t_image)
                im_wmrk = cv2.imread(t_image).astype(np.single)
    
            extracted_wm = extract_watermark(t_image, tile_size=tl, how=w_method)
              
            v = 0
            extr_w = None
            for ii in range(len(extracted_wm)):
                extr_w_ = "".join(["1" if extracted_wm[ii][i] else "0" for i in range(len(extracted_wm[ii]))])
                v_ = np.sum(np.asarray(wm) == np.asarray(extracted_wm[ii])) / l
                if v_ > v:
                    v = v_
                    extr_w = extr_w_    
        
            qt[full_attack_name + " - " + str(q)] = {'msg': extr_w, 'correct bits': v}
        
            print(f'{str(q)}.[{full_attack_name}]("{k}"): retrieved message = {extr_w}; {"Pass" if (v == 1) else "Failed"}')

            os.remove(t_image)
                    
        qv[k][w]['validation'] = qt

        with open(out_file, 'wb') as f:
            pickle.dump(qv, f)          

    if os.path.isfile('input_image.png'): os.remove('input_image.png')
        
