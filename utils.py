import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
#from tensorflow.keras.layers import *
import pickle
import math
#from tensorflow.keras import backend as K
#from IPython.display import clear_output
import sys
import cv2
import torch
import dgl

max_num_node = 24
canvas_size = 550

flip_bird = [1,3,2,4,5,6,8,7,11,12,9,10,13]
flip_cow = [1,3,2,5,4,6,8,7,9,10,13,14,11,12,17,18,15,16,19]
flip_cat = [1,3,2,5,4,6,7,8,11,12,9,10,15,16,13,14,17]
flip_dog = [1,3,2,5,4,6,7,8,11,12,9,10,15,16,13,14,17,18]
flip_horse = [1,3,2,5,4,6,8,7,9,10,13,14,11,12,17,18,15,16,19,21,20]
flip_person = [1,3,2,5,4,7,6,8,9,10,11,12,16,17,18,13,14,15,22,23,24,19,20,21]
flip_sheep = flip_cow

person_tree = {}
person_tree[0] = [1,2,3,4,7,8,9,10,11]
person_tree[1] = [0,2,3,5,7]
person_tree[2] = [0,1,4,6,7]
person_tree[3] = [0,1]
person_tree[4] = [0,2]
person_tree[5] = [1]
person_tree[6] = [2]
person_tree[7] = [0,1,2,8]
person_tree[8] = [0,7]
person_tree[9] = [0]
person_tree[10] = [0,11,13,16,19,22]
person_tree[11] = [0,10]
person_tree[12] = [13,14]
person_tree[13] = [10,12]
person_tree[14] = [12]
person_tree[15] = [16,17]
person_tree[16] = [10,15]
person_tree[17] = [15]
person_tree[18] = [19,20]
person_tree[19] = [18,10]
person_tree[20] = [18]
person_tree[21] = [22,23]
person_tree[22] = [21,10]
person_tree[23] = [21]

bird_tree = {}
bird_tree[0] = [1,2,3,4,5]
bird_tree[1] = [0,2,3]
bird_tree[2] = [0,1,3]
bird_tree[3] = [0,1,2]
bird_tree[4] = [0,5,6,7,8,10,12]
bird_tree[5] = [0,4]
bird_tree[6] = [7,4]
bird_tree[7] = [6,4]
bird_tree[8] = [4,9]
bird_tree[9] = [8]
bird_tree[10] = [11,4]
bird_tree[11] = [10]
bird_tree[12] = [4]

dog_tree = {}
dog_tree[0] = [1,2,3,4,5,6,7,17]
dog_tree[1] = [2,0,3]
dog_tree[2] = [0,1,4]
dog_tree[3] = [0,1]
dog_tree[4] = [0,2]
dog_tree[5] = [0,1,2]
dog_tree[6] = [0,8,10,12,14,16,7]
dog_tree[7] = [0,6]
dog_tree[8] = [9,6]
dog_tree[9] = [8]
dog_tree[10] = [6,11]
dog_tree[11] = [10]
dog_tree[12] = [13,6]
dog_tree[13] = [12]
dog_tree[14] = [6,15]
dog_tree[15] = [14]
dog_tree[16] = [6]
dog_tree[17] = [0]

cat_tree = {}
cat_tree[0] = [1,2,3,4,5,6,7]
cat_tree[1] = [2,0,3]
cat_tree[2] = [0,1,4]
cat_tree[3] = [0,1]
cat_tree[4] = [0,2]
cat_tree[5] = [0,1,2]
cat_tree[6] = [0,8,10,12,14,16,7]
cat_tree[7] = [0,6]
cat_tree[8] = [9,6]
cat_tree[9] = [8]
cat_tree[10] = [6,11]
cat_tree[11] = [10]
cat_tree[12] = [13,6]
cat_tree[13] = [12]
cat_tree[14] = [6,15]
cat_tree[15] = [14]
cat_tree[16] = [6]

horse_tree = {}
horse_tree[0] = [1,2,3,4,5,8,9]
horse_tree[1] = [2,0,3]
horse_tree[2] = [0,1,4]
horse_tree[3] = [0,1]
horse_tree[4] = [0,2]
horse_tree[5] = [0,1,2]
horse_tree[6] = [11] #lfho
horse_tree[7] = [13] #rfho
horse_tree[8] = [0,10,12,14,16,18]
horse_tree[9] = [0,8]
horse_tree[10] = [8,11,12]
horse_tree[11] = [10,6]
horse_tree[12] = [10,8,13]
horse_tree[13] = [7]
horse_tree[14] = [8,15,16]
horse_tree[15] = [14,19]
horse_tree[16]= [14,17]
horse_tree[17]= [16,20]
horse_tree[18]= [8]
horse_tree[19] = [15]
horse_tree[20] = [17]

cow_tree = {}
cow_tree[0] = [1,2,3,4,5,6,7,8,9]
cow_tree[1] = [2,0,3,5]
cow_tree[2] = [0,1,4,5]
cow_tree[3] = [0,1,6]
cow_tree[4] = [0,2,7]
cow_tree[5] = [0,1,2]
cow_tree[6] = [0,3] #lfho
cow_tree[7] = [0,4,13] #rfho
cow_tree[8] = [0,9,10,12,14,16,18]
cow_tree[9] = [0,8]
cow_tree[10] = [8,11,12]
cow_tree[11] = [10,6]
cow_tree[12] = [10,8,13]
cow_tree[13] = [7,12]
cow_tree[14] = [8,15,16]
cow_tree[15] = [14]
cow_tree[16]= [8,14,17]
cow_tree[17]= [16]
cow_tree[18]= [8]

motorbike_tree = {}
motorbike_tree[0] = [14,1,2]
motorbike_tree[1] = [14,0]
motorbike_tree[2] = [14,0]
motorbike_tree[3] = [14]
motorbike_tree[4] = [14]
motorbike_tree[5] = [14]
motorbike_tree[6] = [14]
motorbike_tree[7] = [14]
motorbike_tree[8] = [14]
motorbike_tree[9] = [14]
motorbike_tree[10]= [14]
motorbike_tree[11]= [14]
motorbike_tree[12]= [14]
motorbike_tree[13]= [14]
motorbike_tree[14]= [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

bicycle_tree = {}
bicycle_tree[0] = [15,1,3]
bicycle_tree[1] = [15,0,2,4]
bicycle_tree[2] = [15,1]
bicycle_tree[3] = [15,0]
bicycle_tree[4] = [15,1]
bicycle_tree[5] = [15]
bicycle_tree[6] = [15]
bicycle_tree[7] = [15]
bicycle_tree[8] = [15]
bicycle_tree[9] = [15]
bicycle_tree[10]= [15]
bicycle_tree[11]= [15]
bicycle_tree[12]= [15]
bicycle_tree[13]= [15]
bicycle_tree[14]= [15]
bicycle_tree[15]= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

aeroplane_tree = {}
aeroplane_tree[0] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
aeroplane_tree[1] = [0]
aeroplane_tree[2] = [0,3]
aeroplane_tree[3] = [0,2]
aeroplane_tree[4] = [0]
aeroplane_tree[5] = [0]
aeroplane_tree[6] = [0]
aeroplane_tree[7] = [0]
aeroplane_tree[8] = [0]
aeroplane_tree[9] = [0]
aeroplane_tree[10]= [0]
aeroplane_tree[11]= [0]
aeroplane_tree[12]= [0]
aeroplane_tree[13]= [0]
aeroplane_tree[14]= [0]
aeroplane_tree[15]= [0]
aeroplane_tree[16]= [0]
aeroplane_tree[17]= [0]
aeroplane_tree[18]= [0]
aeroplane_tree[19]= [0]
aeroplane_tree[20]= [0]
aeroplane_tree[21]= [0]
aeroplane_tree[22]= [0]

tree = { 'aeroplane':aeroplane_tree, 'motorbike':motorbike_tree,'bicycle':bicycle_tree, 'person':person_tree, 'cow':cow_tree, 'dog':dog_tree, 'cat':cat_tree, 'sheep':cow_tree, 'bird':bird_tree, 'horse':horse_tree }

object_names = ['cow','sheep','bird','person','cat','dog','horse','aeroplane','motorbike','bicycle']

class_dic = {'cow':0,'sheep':1,'bird':2,'person':3,'cat':4,'dog':5,'horse':6,'aeroplane':7,'motorbike':8,'bicycle':9,'car':10}

def get_pos(bbx):
    temp_pos = []
    for i in bbx:
        if i.tolist()!=[0,0,0,0]:
            temp_pos.append([1])
        elif i.tolist()==[0,0,0,0]:
            temp_pos.append([0])

    return np.asarray(temp_pos)

#colors = [(229,184,135), (0,0,255), (0,255,0),(255,0,0),(0,255,255),(255,255,0),(255,0,255),(130,0,75),(0,128,128),(128,128,0),(128,128,128),(0,0,0),(30,105,210),(30,105//2,210//2),(180,105,255),(180//2,105//2,255),(100,100,30),(0,100//2,20),(128,0,128),(30,105,210),(255//2,105,255),(180//2,105,255//2),(50,100,0), (229//2,184,135//2),(229,184,135), (0,0,255), (0,255,0),(255,0,0),(0,255,255),(255,255,0),(255,0,255),(130,0,75),(0,128,128),(128,128,0),(128,128,128),(0,0,0),(30,105,210),(30,105//2,210//2),(180,105,255),(180//2,105//2,255),(100,100,30),(0,100//2,20),(128,0,128),(30,105,210),(255//2,105,255),(180//2,105,255//2),(50,100,0), (229//2,184,135//2)]
colors = [(1, 0, 0),
          (0.737, 0.561, 0.561),
          (0.255, 0.412, 0.882),
          (0.545, 0.271, 0.0745),
          (0.98, 0.502, 0.447),
          (0.98, 0.643, 0.376),
          (0.18, 0.545, 0.341),
          (0.502, 0, 0.502),
          (0.627, 0.322, 0.176),
          (0.753, 0.753, 0.753),
          (0.529, 0.808, 0.922),
          (0.416, 0.353, 0.804),
          (0.439, 0.502, 0.565),
          (0.784, 0.302, 0.565),
          (0.867, 0.627, 0.867),
          (0, 1, 0.498),
          (0.275, 0.51, 0.706),
          (0.824, 0.706, 0.549),
          (0, 0.502, 0.502),
          (0.847, 0.749, 0.847),
          (1, 0.388, 0.278),
          (0.251, 0.878, 0.816),
          (0.933, 0.51, 0.933),
          (0.961, 0.871, 0.702)]
colors = (np.asarray(colors)*255)


def arrangement(a, b, object_name):
    if object_name=='cow' or object_name=='sheep':
        p = [ 10,11,18,13,12,14,16,15,17,9,0,7,3,4,5,6,1,2,8 ]
    elif object_name=='bird':
        p = [ 10,11,12,9,0,7,3,4,5,6,1,2,8 ]
    elif object_name=='person':
        p = [ 10,11,19,18,20,22,21,23,13,12,14,16,15,17,9,0,7,3,4,5,6,1,2,8 ]
    elif object_name=='cat':
        p = [ 10,11,13,12,14,16,15,9,0,7,3,4,5,6,1,2,8]
    elif object_name=='dog':
        p = [ 10,11,13,12,14,16,15,17,9,0,7,3,4,5,6,1,2,8]
    elif object_name=='horse':
        p = [ 10,11,19,18,20,13,12,14,16,15,17,9,0,7,3,4,5,6,1,2,8 ]
    elif object_name=='aeroplane':
        p = [ 10,11,19,18,20,22,21,13,12,14,16,15,17,9,0,7,3,4,5,6,1,2,8 ]
    elif object_name=='car':
        p = [ 10,11,19,18,20,22,21,23,24,25,26,27,28,13,12,14,16,15,17,9,0,7,3,4,5,6,1,2,8 ]
    elif object_name=='motorbike':
        p = [ 10,11,13,12,14,9,0,7,3,4,5,6,1,2,8 ]
    elif object_name=='bicycle':
        p = [ 10,11,13,12,14,15,9,0,7,3,4,5,6,1,2,8 ]
    else:
      print("error")
    return a[p], b[p]

def rearrange(lbl, bbx, mask, object_name):
    if object_name=='cow' or object_name=='sheep':
        p = np.asarray([1,3,2,5,4,6,8,7,9,10,13,14,11,12,17,18,15,16,19])-1
    elif object_name=='bird':
        p = np.asarray([1,3,2,4,5,6,8,7,11,12,9,10,13])-1
    elif object_name=='person':
        p = np.asarray([1,3,2,5,4,7,6,8,9,10,11,12,16,17,18,13,14,15,22,23,24,19,20,21])-1
    elif object_name=='cat':
        p = np.asarray([1,3,2,5,4,6,7,8,11,12,9,10,15,16,13,14,17])-1
    elif object_name=='dog':
        p = np.asarray([1,3,2,5,4,6,7,8,11,12,9,10,15,16,13,14,17,18])-1
    elif object_name=='horse':
        p = np.asarray([1,3,2,5,4,6,8,7,9,10,13,14,11,12,17,18,15,16,19,21,20])-1
    elif object_name=='aeroplane':
        p = np.asarray([1,3,2,5,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])-1
    elif object_name=='car':
        p = np.asarray([1,3,2,4,5,7,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])-1
    elif object_name=='motorbike':
        p = np.asarray([1,3,2,4,5,7,6,8,9,10,11,12,13,14,15])-1
    elif object_name=='bicycle':
        p = np.asarray([1,3,2,4,5,7,6,8,9,10,11,12,13,14,15,16])-1
    else:
      print("error")
    return lbl[p], bbx[p], mask[p]

def pad_along_axis(array, target_length, axis=0):
    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)
    if pad_size < 0:
        return array

    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return b

def bounder(img):
    result = np.where(img<0.5)
    listOfCoordinates = list(zip(result[0], result[1]))
    for cord in listOfCoordinates:
        img[cord] = 0
    result1 = np.where(img>=0.5)
    listOfCoordinates1 = list(zip(result1[0], result1[1]))
    for cord in listOfCoordinates1:
        img[cord] = 1
    return img

def add_images(canvas,img, ii):
    result = np.where(img!=0)
    listOfCoordinates = list(zip(result[0], result[1]))
    for cord in listOfCoordinates:
        canvas[cord] = ii
    return canvas

label_to_color = {0:(0,0,0),
                    1:(0.941, 0.973, 1),
                    2:(0.98, 0.922, 0.843),
                    3:(0, 1, 1),
                    4:(0.498, 1, 0.831),
                    5:(0.941, 1, 1),
                    6:(0.961, 0.961, 0.863),
                    7:(1, 0.894, 0.769),
                    8:(0.251, 0.878, 0.816),
                    9:(1, 0.388, 0.278),
                    10:(0, 0, 1),
                    11:(0.541, 0.169, 0.886),
                    12:(0.647, 0.165, 0.165),
                    13:(0.871, 0.722, 0.529),
                    14:(0.373, 0.62, 0.627),
                    15:(0.498, 1, 0),
                    16:(0.824, 0.412, 0.118),
                    17:(1, 0.498, 0.314),
                    18:(0.392, 0.584, 0.929),
                    19:(0.275, 0.51, 0.706),
                    20:(0.863, 0.0784, 0.235),
                    21:(0, 1, 1),
                    22:(0, 0, 0.545),
                    23:(0.824, 0.706, 0.549),
                    24:(0.251, 0.878, 0.816)}

def label_2_image(img):
    rgb_img = np.zeros((img.shape[0],img.shape[1], 3))
    for key in label_to_color.keys():
        rgb_img[img == key] = label_to_color[key]
    return rgb_img

def make_mask(box,mask):
    b_in = np.copy(box)
    mx = np.copy(mask)
    max_parts = len(box)
    xmax = max(box[:,2])
    ymax = max(box[:,3])
    canvas = np.zeros((int(ymax),  int(xmax)), np.float32)
    b_in, mx = arrangement(b_in, mx,object_name)
    for i in range(max_parts):
        x_min, y_min, x_max, y_max = b_in[i]
        if x_max-x_min > 0 and y_max-y_min>0:
            x, y = canvas[ int(y_min):int(y_max), int(x_min):int(x_max) ].shape
            canvas[ int(y_min):int(y_max), int(x_min):int(x_max) ] = add_images(canvas[ int(y_min):int(y_max), int(x_min):int(x_max)  ],cv2.resize(bounder(np.squeeze(mx[i]))*(i+1), (y,x)), i+1)
    plt.imshow(label_2_image(canvas))
    plt.show()
    return label_2_image(canvas)

def plot_image_bbx(bbx,image):
    canvas = np.copy(image)
    i = 0
    for coord in bbx:
        x_minp, y_minp,x_maxp , y_maxp= coord
        if [x_minp, y_minp,x_maxp , y_maxp]!=[0,0,0,0]:
            cv2.rectangle(canvas, ((x_minp), (y_minp)), ((x_maxp) , (y_maxp) ), colors[i], 4)
        i = i+1
    plt.imshow(canvas)
    plt.show()
    return canvas

def flip_mask(mask):
    mx = np.copy(mask)
    for i in range(len(mx)):
        mx[i] = mx[i][:,::-1]
    return mx

def flip_bbx(label, bbx, img):
    bx = np.copy(bbx)
    x_min = min(bbx[:,0])
    y_min = min(bbx[:,1])
    x_max = max(bbx[:,2])
    y_max = max(bbx[:,3])
    img_center = np.asarray( [((x_max+x_min)/2),  ((y_max+y_min)/2)] )
    img_center = np.hstack( (img_center, img_center) )
    bx[:,[0,2]] += 2*(img_center[[0,2]] - bx[:,[0,2]])
    box_w = abs(bx[:,0] - bx[:,2])
    bx[:,0] -= box_w
    bx[:,2] += box_w
    for i in range(len(label)):
        if sum(label[i])==0:
            bx[i][0] = 0
            bx[i][1] = 0
            bx[i][2] = 0
            bx[i][3] = 0
    return bx

def flip_data_instance(label, box, mask, image,object_name):
    bx = np.copy(flip_bbx(label,box,image))
    mx = np.copy(flip_mask(mask))
    ix = np.copy(image[:,::-1])
    lx = np.copy(label)
    lx, bx, mx = rearrange(lx, bx, mx,object_name)
    return lx,bx,mx,ix
def cordinates(img):
    y_min = 0
    y_max = 0
    x_min = 0
    x_max = 0

    for i in img:
        if np.count_nonzero(i) is not 0:
            break
        y_min+=1

    for i in img.T:
        if np.count_nonzero(i) is not 0:
            break
        x_min+=1

    for i in img[::-1]:
        if np.count_nonzero(i) is not 0:
            break
        y_max+=1
    y_max = img.shape[0] - y_max - 1

    for i in img.T[::-1]:
        if np.count_nonzero(i) is not 0:
            break
        x_max+=1
    x_max = img.shape[1] - x_max - 1

    return x_min, y_min, x_max, y_max

def rotate_im(image, angle):
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image

def get_corners(bboxes):

    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)

    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)

    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))

    return corners

def clip_box(bbox, clip_box, alpha):
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
    y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
    x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
    y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))

    delta_area = ((ar_ - bbox_area(bbox))/ar_)

    mask = (delta_area < (1 - alpha)).astype(int)

    bbox = bbox[mask == 1,:]


    return bbox

def rotate_box(corners,angle,  cx, cy, h, w):

    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)


    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T

    calculated = calculated.reshape(-1,8)

    return calculated

def get_enclosing_box(corners):
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]

    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)

    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))

    return final

def bbox_area(bbox):
    return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])

def rtt(angle, label, img, bboxes):

    w,h = img.shape[1], img.shape[0]
    cx, cy = w//2, h//2

    img = rotate_im(img, angle)

    corners = get_corners(bboxes)

    corners = np.hstack((corners, bboxes[:,4:]))


    corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)

    new_bbox = get_enclosing_box(corners)


    scale_factor_x = img.shape[1] / w

    scale_factor_y = img.shape[0] / h

    img = cv2.resize(img, (w,h))

    new_bbox[:,:4] = np.true_divide(new_bbox[:,:4], [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y])

    for i in range(len(label)):
        if sum(label[i])==0:
            new_bbox[i][0] = 0
            new_bbox[i][1] = 0
            new_bbox[i][2] = 0
            new_bbox[i][3] = 0

    return img, new_bbox
def render_mask(box,mask,angle):
    mx = np.copy(mask)
    b_in = np.copy(box)
    max_parts = len(box)
    xmax = max(box[:,2])
    ymax = max(box[:,3])
    temp_mx_list = []
    temp_bx_list = []
    for i in range(max_parts):
        canvas = np.zeros((int(ymax),  int(xmax)), np.float32)
        x_min, y_min, x_max, y_max = b_in[i]
        if x_max-x_min > 0 and y_max-y_min>0:

            x, y = canvas[ int(y_min):int(y_max), int(x_min):int(x_max) ].shape
            canvas[ int(y_min):int(y_max), int(x_min):int(x_max) ] = add_images(canvas[ int(y_min):int(y_max), int(x_min):int(x_max)  ],cv2.resize(bounder(np.squeeze(mx[i]))*(i+1), (y,x)), i+1)
            canvas = rotate_im(canvas,angle)
            x_min, y_min, x_max, y_max = cordinates(canvas)
            #canvas = canvas[int(y_min):int(y_max), int(x_min):int(x_max)]
            #resized_cropped = np.expand_dims(cv2.resize(canvas, (64, 64)), axis = 3)
        temp_bx_list.append([x_min, y_min, x_max, y_max])
        #temp_mx_list.append(resized_cropped)
        #plt.imshow(canvas)
        #plt.show()
    return np.asarray(temp_bx_list,dtype="float32")
def scale(bbx, scaling_factor):

    height = max(bbx[:,3])
    width = max(bbx[:,2])

    pos = get_pos(bbx)

    fold_a = np.copy(bbx)
    fold_b = np.copy(bbx)
    fold_c = np.copy(bbx)
    fold_d = np.copy(bbx)

    scale_height = scaling_factor
    scale_width = scaling_factor

    fold_a[:,0] = (fold_a[:,0]-scale_width)
    fold_b[:,1] = (fold_b[:,1]-scale_height)
    fold_c[:,2] = (fold_c[:,2]+scale_width)
    fold_d[:,3] = (fold_d[:,3]+scale_height)

    return fold_a*pos,fold_b*pos,fold_c*pos,fold_d*pos

def centre_object(bbx,canvas_size):

    pos = get_pos(bbx)
    bx = np.copy(bbx)

    h,w = canvas_size

    h_o = max(bbx[:,3])
    w_o = max(bbx[:,2])

    h_shift = int(h/2 - h_o/2)
    w_shift = int(w/2 - w_o/2)

    bx[:,0] = (bx[:,0]+w_shift)
    bx[:,1] = (bx[:,1]+h_shift)
    bx[:,2] = (bx[:,2]+w_shift)
    bx[:,3] = (bx[:,3]+h_shift)

    return bx*pos

def append_labels(box): # appends part presence information
                        # along with the bbox submatrix
  all_box = []
  for bbx in box:
    pos = get_pos(bbx)
    bbx = (((bbx/canvas_size)))*pos #scaling bbox coords btw 0 and 1

    #generating part presence vector and appending in front of bbox coords
    temp = []
    for bx in bbx:
      if bx.tolist()!=[0,0,0,0]:
        temp.append([1]+bx.tolist())
      else:
        temp.append([0]+bx.tolist())
    all_box.append(temp)
  return np.asarray(all_box)

def plot_bbx(bbx):
    canvas = np.ones((canvas_size,canvas_size,3), np.uint8) * 255
    for i, coord in enumerate(bbx):
        x_minp, y_minp,x_maxp , y_maxp= coord
        if [x_minp, y_minp,x_maxp , y_maxp]!=[0,0,0,0]:
            cv2.rectangle(canvas, (int(x_minp), int(y_minp)), (int(x_maxp) , int(y_maxp) ), colors[i], 6)
    return canvas

def transform_bbx(bbx1):

    eps = 0.00001
    bbx = np.copy(bbx1)
    bxx = np.copy(bbx)

    bbx[:,0] = np.exp(bbx[:,0])
    bbx[:,1] = np.exp(bbx[:,1])
    bbx[:,2] = np.exp(bbx[:,2])
    bbx[:,3] = np.exp(bbx[:,3])

    bxx[:,0] = bbx[:,0]
    bxx[:,1] = bbx[:,1]
    bxx[:,2] = bbx[:,0] + (bbx[:,3])
    bxx[:,3] = bbx[:,1] + (bbx[:,2])

    return

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L

def compute_ciou(target,  output):

    #output = ((output)*(target != 0))
    #target = ((target)*(target != 0))

    #x1g, y1g, x2g, y2g = tf.split(value=target, num_or_size_splits=4, axis=-1)
    #x1, y1, x2, y2 = tf.split(value=output, num_or_size_splits=4, axis=-1)

    x1g, y1g, x2g, y2g = target[:,:,0], target[:,:,1], target[:,:,2], target[:,:,3]
    x1, y1, x2, y2 = output[:,:,0], output[:,:,1], output[:,:,2], output[:,:,3]

    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g

    x_center = (x2 + x1) / 2
    y_center = (y2 + y1) / 2
    x_center_g = (x1g + x2g) / 2
    y_center_g = (y1g + y2g) / 2

    xc1 = torch.min(x1,x1g)
    yc1 = torch.min(y1,y1g)
    xc2 = torch.min(x2,x2g)
    yc2 = torch.min(y2,y2g)

    #iou
    xA = torch.max(x1g, x1)
    yA = torch.max(y1g, y1)
    xB = torch.min(x2g, x2)
    yB = torch.min(y2g, y2)

    zero = torch.zeros(xA.shape)

    interArea = torch.max(zero, (xB - xA + 1)) * torch.max(zero, yB - yA + 1)
    print('interArea.shape',interArea.shape)
    boxAArea = (x2g - x1g +1) * (y2g - y1g +1)
    print('bboxA',boxAArea.shape)
    boxBArea = (x2 - x1 +1) * (y2 - y1 +1)
    print('bboxB',boxBArea.shape)
    iouk = interArea / (boxAArea + boxBArea - interArea)
    print('iouk',iouk.shape)
    ciouk = -torch.log(iouk) #+ u




    ##

    #iouk = iou(target,  output)

    ##distance###
    #c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + 1e-7
    #d = ((x_center - x_center_g) ** 2) + ((y_center - y_center_g) ** 2)
    #u = d / c
    ##
    ##aspect ratio###
    #arctan = tf.atan(w_gt/(h_gt + 1e-7))-tf.atan(w_pred/(h_pred + 1e-7))
    #v = (4 / (math.pi ** 2)) * tf.pow((tf.atan(w_gt/(h_gt + 1e-7))-tf.atan(w_pred/(h_pred + 1e-7))),2)
    #S = 1 - iouk
    #alpha = v / (S + v + 1e-7)
    #w_temp = 2 * w_pred
    #ar = (8 / (math.pi ** 2)) * arctan * ((w_pred - w_temp) * h_pred)
    ###

    #ciouk = -tf.log(iouk) + (u + alpha * ar)

    #ciouk = (1 - ciouk)
    #ciouk = tf.where(tf.is_nan(ciouk), tf.zeros_like(ciouk), ciouk)
    return torch.mean(ciouk)

def box_loss(tru_box, gen_box):
    #tru_box = (tf.log(tru_box))
    #tru_box = tf.where(tf.math.is_inf(tru_box), tf.ones_like(tru_box) * 0, tru_box)

    #gen_box = (tf.log((gen_box)*(tru_box != 0)))
    #gen_box = tf.where(tf.math.is_inf(gen_box), tf.ones_like(gen_box) * 0, gen_box)
    #gen_box = ((gen_box)*(tru_box != 0))
    #tru_box = ((tru_box)*(tru_box != 0))
    #sum_r = tf.dtypes.cast(tf.reduce_sum(tf.keras.losses.MSE(tru_box, gen_box)), tf.float32)
    #mseloss1 = torch.nn.MSELoss(reduction='none')
    mseloss2 = torch.nn.MSELoss()
    #mloss = mseloss1(gen_box,tru_box)
    #mloss2 = mseloss2(gen_box,tru_box)
    #print('shape of mloss2: ',mloss2.shape,mloss2)
    #print('shape of mloss: ',mloss)
    sum_r = mseloss2(gen_box,tru_box).type('torch.FloatTensor')
    print('Sum_r: ',sum_r)
    #num_r = tf.dtypes.cast(tf.math.count_nonzero(tf.reduce_sum(tf.keras.losses.MSE(tru_box, gen_box), axis=-1)), tf.float32)
    #f = mseloss1(gen_box,tru_box)
    #print('f',f.shape,type(f))
    #num_r = torch.nonzero(mseloss1(gen_box,tru_box)).type('torch.FloatTensor')
    #print('bloss',sum_r,num_r.shape,num_r)
    #num_r = num_r.shape[0]
    #return (sum_r/(num_r+1))
    return sum_r

def add_self_loop(g):
    """Return a new graph containing all the edges in the input graph plus self loops
    of every nodes.
    No duplicate self loop will be added for nodes already having self loops.
    Self-loop edges id are not preserved. All self-loop edges would be added at the end.

    Examples
    ---------

    >>> g = DGLGraph()
    >>> g.add_nodes(5)
    >>> g.add_edges([0, 1, 2], [1, 1, 2])
    >>> new_g = dgl.transform.add_self_loop(g) # Nodes 0, 3, 4 don't have self-loop
    >>> new_g.edges()
    (tensor([0, 0, 1, 2, 3, 4]), tensor([1, 0, 1, 2, 3, 4]))

    Parameters
    ------------
    g: DGLGraph

    Returns
    --------
    DGLGraph
    """
    new_g = DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    src, dst = g.all_edges(order="eid")
    src = F.zerocopy_to_numpy(src)
    dst = F.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    return new_g
