import math, cv2, torchvision
import numpy as np


def pad_rgbarray_to_tensor(rgb_img, multiple=1): 
    # pad frame such that each dimension is multiples of "multiple"
    H,W,C = rgb_img.shape
    imh = math.ceil(H/multiple)*multiple
    imw = math.ceil(W/multiple)*multiple
#     print('Transforming image from {}x{} to {}x{}...'.format(W, H, imw, imh))
    
    img_transforms = torchvision.transforms.Compose([
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Pad([(imw - W)//2, 
                                             (imh - H)//2],
                                            padding_mode='symmetric'
                                           )
    ])
    rgb_tensor = img_transforms(rgb_img).float()   
    return rgb_tensor

def tensor_to_rgbarray(tensor): 
    '''
    [0,1] -> [0,255]
    [C, H, W] -> [H, W, C]
    '''
    img = (np.moveaxis(tensor.numpy(), 0, -1) * 255).astype('uint8')
    # weird hack to convert from numpy to opencv umat 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
#     return (np.moveaxis(tensor.numpy(), 0, -1) * 255).astype('uint8')