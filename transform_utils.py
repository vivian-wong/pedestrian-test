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

def image_to_world(p, Homog):
    '''
    p: a np array of [[x1, y1], [x2, y2], ... ]; shape (N,2)
    '''
    # from Opentraj repo
    pp = np.stack((p[:, 0], p[:, 1], np.ones(len(p))), axis=1) # append a rightmost column of 1s
    PP = np.matmul(Homog, pp.T).T # [world x, world y, 1] (3 x 3)
    P_normal = PP / np.repeat(PP[:, 2].reshape((-1, 1)), 3, axis=1)  # normalize since the third column is not exactly 1??
    return P_normal[:, :2]*0.8 # not sure why *0.8 but that's in the code. 

# #### converting any coordinate to world coordinate 
# def img_to_world_coords(coords_img, HOMOG, pts_img = None, pts_world = None):
#     '''
#     use at least 4 
#     input: 
#         coords_img: a np array of dim (num_ped,seq_len,2)
#     if using pre-computed HOMOG, no need to use pts_img and pts_world
#         pts_img: a list of [x,y] points in image coordinates. list length cannot be shorter than 4
#         pts_world: a list of [x,y] points in world coordinates. each point cooresponds to a point in pts_img
#         assumed_height: the estimated height of all humans. source: https://www.sciencedirect.com/science/article/pii/S0379073811002167
#                         NOT USED FOR NOW 
#     output: 
#         coords_world: same shape as coords_img but in world coordinates
#     '''
#     # convert to np matrix and check size
#     if pts_img == None: 
#         h = HOMOG
#     else: 
#         pts_img = np.array(pts_img)
#         pts_world = np.array(pts_world)
#         assert pts_img.shape==pts_world.shape and pts_img.shape[0] >= 4
     
#         # calculate homography matrix H. more pts_img means more accurate H
#         h, status = cv2.findHomography(pts_img, pts_world)
     
#     # finally, get the mapped world coordinates
#     coords_world = cv2.perspectiveTransform(coords_img.astype(np.float32), h)
#     # coords_world[:,1] += assumed_height
#     return coords_world