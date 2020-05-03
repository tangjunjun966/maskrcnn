"""
MASKRCNN algrithm for object detection and instance segmentation
Written and modified by tang jun on JAN , 2019
if you  have  questions , please connect me by Email： tangjunjunfighter@163.com
"""



import scipy
# import os
# import random
# import datetime
# import re
# import math
# import logging
# from collections import OrderedDict
# import multiprocessing
# import numpy as np
import tensorflow as tf
import keras
# import keras.backend as K  # keras中的后端backend及其相关函数
# import keras.layers as KL
# import keras.engine as KE
# import keras.models as KM



import math
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
from PIL import Image
import random
# from mrcnn1 import utils, model as modellib, visualize
# from mrcnn1 import utils, model as modellib, visualize
import model as modellib
# from  mrcnn1 import visualize

from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


ROOT_DIR = os.getcwd()  # 得到当前路径
sys.path.append(ROOT_DIR)  # To find local version of the library
# Directory to save logs and trained models
MODEL_DIR = os.path.join(ROOT_DIR, "logs")  # 在当前路径的logs文件路径
iter_num = 0
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")  # 载入训练模型权重路径


class Config_config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 256
    NUM_CLASSES = 1 + 4  # Override in sub-classes
    PRE_NMS_LIMIT = 6000  # 判断在训练时候的提取层提取个数，若大于anchors则提取anchors个，否则相反
    IMAGE_CHANNEL_COUNT = 3


    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    # NAME = "shapes"  # Override in sub-classes
    # GPU_COUNT = 1
    # IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 5

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    # COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256  # 定义rpn后每一层的通道数

    # Number of classification classes (including background)


    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7  # 小于该阈值被保留

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256  # rpn数据需要此值，rpn网络也需要次之

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000   # 训练模型在rpn后提取层的数量
    POST_NMS_ROIS_INFERENCE = 1000  # 测试模型在rpn后提取层的数量

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and inferencing
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"

    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # Howver, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 100  # target层

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when inferencing
    TRAIN_BN = True  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0
    batch_size=1

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        # self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, 3])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
# 预测图片基本配置更改
class Predict_Config(Config_config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 256
    batch_size = 1


config = Config_config()  # 基本配置建立实列
config.display()  # 显示基本配置


import skimage.color
import skimage.io
import skimage.transform
class Dataset_data(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self.image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{ "id": 0, "name": "BG"}]
        # self.source_class_ids = {"":[0],"shapes": [0,1,2,3,4]}
        self.class_names = []  # 包含0背景名字


    def add_class(self,  class_id, class_name):
        # assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            # "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self,  image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            # "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def data_load_information(self, img_floder):  # count表示transon文件的数量 img_floder 是transon文件路径
        """
        该函数向class_info添加不良类的代码必须手动修改，
        该函数主要保存类别信息，图片信息（如原始图片路径，
        高宽及mask图片路径等）。
        该函数只要输入文件名字，它会自动遍历所有文件，
        并保存文件图片的信息。
        """
        # Add classes
        self.add_class( 1, "line_bulge")  # 添加标签,这里只有一个不良  ###########################################################
        self.add_class( 2, "dot_concave")
        self.add_class( 3, "dot_bulge")
        self.add_class( 4, "Irregular_concave")
        img_file_list = os.listdir(img_floder)  # 返回文件夹中包含的名字目录
        count = len(img_file_list)  # 有多少数量
        id = 0
        for sorce_path in img_file_list:  # 遍历所有文件夹
            yaml_path = os.path.join(img_floder + '\\' + sorce_path, 'info.yaml')  # label_names: - _background_  - NG
            mask_path = os.path.join(img_floder + '\\' + sorce_path, 'label.png')
            img_path = os.path.join(img_floder + '\\' + sorce_path, 'img.png')
            cv_img = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8),
                                  cv2.IMREAD_UNCHANGED)  # np.fromfile以np.uint8读取文件  # cv2.imdecode缓存中读取数据，并解码成图像格式
            self.add_image( image_id=id, path=img_path, width=cv_img.shape[1], height=cv_img.shape[0],
                           mask_path=mask_path, yaml_path=yaml_path)
            id += 1
            if id > count:
                break
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [c["name"] for c in self.class_info]  # 保存图片类别，包含0
        self.num_images = len(self.image_info)  # 保存图片数量
        self.image_ids = np.arange(self.num_images)  # 根据图片数量产生图片编号


    def load_image(self, image_id):
        """
        该函数在数据产生时候使用
        Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image


    def load_mask(self, image_id):
        """
        该函数也是在数据产生中使用，主要根据图片序列，产生图片的mask，
        将有mask的修改成值为1，其它为0，并返回每个mask对应的类数值，
        返回mask与class_ids，其中mask为[w,h,object],
        class_ids为[object]，如[w,h,4]与[1,3,1,2]
        """
        # global iter_num
        info = self.image_info[image_id]  # according  image_id that belong int to choose image_info information
        img = Image.open(info['mask_path'])  # loading mask_path from label_image that original image handled have changed mask image with label
        num_obj = np.max(img)  # 取一个最大值,得到验证有多少个物体就会是多少，如这张图有3个mask则该值等于3
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        count=1
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    # info['width'] 与info['height'] 为label.png图像的宽度与高度
                    at_pixel = img.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1  # 将有mask位置取1
        mask = mask.astype(np.uint8)


        # occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)  #
        # for i in range(count - 2, -1, -1):
        #     mask[:, :, i] = mask[:, :, i] * occlusion
        #     occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        #


        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        num_classes=len(self.class_info)  # 包含背景BG
        for i in range(len(labels)):  # search image_id label to add labels_form.append
            for j in range(1,num_classes):
                if labels[i].find(self.class_info[j]["name"]) != -1:  # find（）function checking if having line_bulge,
                # if so ,return start index  if not ,return -1.therefore judge return value equal -1
                    labels_form.append(self.class_info[j]["name"])
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        # 按照class_ids 选定图片，然后按照yaml文件的分类匹配到class中，并给出整数代表
        return mask, class_ids.astype(np.int32)

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        '''
        temp={'label_names': ['_background_', '11111', '22222', '3333']}
        labels=['_background_', '11111', '22222', '3333']
        labels[0]=['11111', '22222', '3333']
        :param image_id:
        :return:
        '''
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    def generate_pyramid_anchors(self, scales, ratios, feature_shapes, feature_strides, anchor_stride):
        """Generate anchors at different levels of a feature pyramid. Each scale
        is associated with a level of the pyramid, but each ratio is used in
        all levels of the pyramid.

        Returns:
        anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
            with the same order of the given scales. So, anchors of scale[0] come
            first, then anchors of scale[1], and so on.
        """
        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        anchors = []
        for i in range(len(scales)):
            # anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i], feature_strides[i], anchor_stride))
            """
            scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
            ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
            shape: [height, width] spatial shape of the feature map over which to generate anchors.
            feature_stride: Stride of the feature map relative to the image in pixels.
            anchor_stride: Stride of anchors on the feature map. For example, if the value is 2 then generate anchors for every other feature map pixel.
            """
            # Get all combinations of scales and ratios
            scale, ratios = np.meshgrid(np.array(scales[i]), np.array(ratios))
            scale = scale.flatten()
            ratios = ratios.flatten()
            shape = feature_shapes[i]
            feature_stride = feature_strides[i]
            # Enumerate heights and widths from scales and ratios
            # 实际得到box的宽与高
            heights = scale / np.sqrt(ratios)
            widths = scale * np.sqrt(ratios)

            # Enumerate shifts in feature space
            # 实际得到box坐标中心
            shifts_y = np.arange(0, shape[0],
                                 anchor_stride) * feature_stride  # anchor_stride 表示原图img/stride缩放后以anchor_stride为步长取像素，
            # 一此作为中心点，而后乘以feature_stride（stride）将像素中心放回原图像位置中。
            shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
            shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

            # Enumerate combinations of shifts, widths, and heights
            box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
            box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

            # Reshape to get a list of (y, x) and a list of (h, w)
            box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
            box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

            # code above make center of bboxes and height width of bboxes

            # Convert to corner coordinates (y1, x1, y2, x2)
            boxes = np.concatenate([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1)
            # convert center height and width coordinate of bbox to four coordinates which respectively represnt  top left corner and lower right corner
            anchors.append(boxes)
        return np.concatenate(anchors, axis=0)

    def resize(self, image, output_shape, order=1, mode='constant', cval=0, clip=True,
               preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
        """A wrapper for Scikit-Image resize().

        Scikit-Image generates warnings on every call to resize() if it doesn't
        receive the right parameters. The right parameters depend on the version
        of skimage. This solves the problem by using different parameters per
        version. And it provides a central place to control resizing defaults.
        """
        if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
            # New in 0.14: anti_aliasing. Default it to False for backward
            # compatibility with skimage 0.13.
            return skimage.transform.resize(
                image, output_shape,
                order=order, mode=mode, cval=cval, clip=clip,
                preserve_range=preserve_range, anti_aliasing=anti_aliasing,
                anti_aliasing_sigma=anti_aliasing_sigma)
        else:
            return skimage.transform.resize(
                image, output_shape,
                order=order, mode=mode, cval=cval, clip=clip,
                preserve_range=preserve_range)


    def resize_image(self,image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
        """Resizes an image keeping the aspect ratio unchanged.

        min_dim: if provided, resizes the image such that it's smaller dimension == min_dim
        max_dim: if provided, ensures that the image longest side doesn't exceed this value.
        min_scale: if provided, ensure that the image is scaled up by at least
            this percent even if min_dim doesn't require it.
        mode: Resizing mode.
            none: No resizing. Return the image unchanged.
            square: Resize and pad with zeros to get a square image of size [max_dim, max_dim].
            pad64: Pads width and height with zeros to make them multiples of 64.
                   If min_dim or min_scale are provided, it scales the image up
                   before padding. max_dim is ignored in this mode.
                   The multiple of 64 is needed to ensure smooth scaling of feature
                   maps up and down the 6 levels of the FPN pyramid (2**6=64).
            crop: Picks random crops from the image. First, scales the image based
                  on min_dim and min_scale, then picks a random crop of
                  size min_dim x min_dim. Can be used in training only.
                  max_dim is not used in this mode.

        Returns:
        image: the resized image
        window: (y1, x1, y2, x2). If max_dim is provided, padding might
            be inserted in the returned image. If so, this window is the
            coordinates of the image part of the full image (excluding
            the padding). The x2, y2 pixels are not included.
        scale: The scale factor used to resize the image
        padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
        """
        # Keep track of image dtype and return results in the same dtype
        image_dtype = image.dtype
        # Default window (y1, x1, y2, x2) and default scale == 1.
        h, w = image.shape[:2]
        window = (0, 0, h, w)
        scale = 1
        padding = [(0, 0), (0, 0), (0, 0)]

        if mode == "none":
            return image, window, scale, padding

        # Scale?
        if min_dim:
            # Scale up but not down
            scale = max(1, min_dim / min(h, w))  # h, w是原始图片的高与宽
        if min_scale and scale < min_scale:  # min_scale是最小填充倍数的，至少要大于它
            scale = min_scale

        # Does it exceed max dim?
        if max_dim and mode == "square":
            image_max = max(h, w)
            if round(image_max * scale) > max_dim:  # 最终原图片最大边扩充不能超过最大max_dim维度，否则重新选择scale
                scale = max_dim / image_max

        # Resize image using bilinear interpolation
        if scale != 1:
            image = self.resize(image, (round(h * scale), round(w * scale)), preserve_range=True)
            # 上一行代码对图像做了resize，那么会改变图像的尺寸，这是我不愿意看到的，我觉的这样会对缺陷特征有损失，
            # 或者出现变异，因此小心这里的变化
        # Need padding or cropping?
        if mode == "square":
            # Get new height and width
            h, w = image.shape[:2]  # 此时已经将原图按照scale进行了改变
            top_pad = (max_dim - h) // 2
            bottom_pad = max_dim - h - top_pad
            left_pad = (max_dim - w) // 2
            right_pad = max_dim - w - left_pad
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)  # 将改变的图片进行了填充
            window = (top_pad, left_pad, h + top_pad, w + left_pad)  # 保存经过resize后图片的真实大小
        elif mode == "pad64":
            h, w = image.shape[:2]
            # Both sides must be divisible by 64
            assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
            # Height
            if h % 64 > 0:
                max_h = h - (h % 64) + 64
                top_pad = (max_h - h) // 2
                bottom_pad = max_h - h - top_pad
            else:
                top_pad = bottom_pad = 0
            # Width
            if w % 64 > 0:
                max_w = w - (w % 64) + 64
                left_pad = (max_w - w) // 2
                right_pad = max_w - w - left_pad
            else:
                left_pad = right_pad = 0
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        else:
            raise Exception("Mode {} not supported".format(mode))
        return image.astype(image_dtype), window, scale, padding

    def resize_mask(self,mask, scale, padding):
        # scale是输入图像的尺寸变化，padding是最大维度的背景填充，mask有效坐标对应原来输入的图像中
        """Resizes a mask using the given scale and padding.
        Typically, you get the scale and padding from resize_image() to
        ensure both, the image and the mask, are resized consistently.

        scale: mask scaling factor
        padding: Padding to add to the mask in the form
                [(top, bottom), (left, right), (0, 0)]
        """
        # Suppress warning from scipy 0.13.0, the output shape of zoom() is
        # calculated with round() instead of int()
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
        # if crop is not None:
        #     y, x, h, w = crop
        #     mask = mask[y:y + h, x:x + w]
        # else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
        return mask

    def extract_bboxes(self,mask):  # [[num_instances, (y1, x1, y2, x2)]]
        # in a word,bbox proced by  mask will contain all mask which value equal 1.
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        # the last dimension for mask （num_instances） is bbox for instance every picture
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([y1, x1, y2, x2])
        return boxes.astype(np.int32)

    def load_image_gt(self, config, image_id, augment=False, augmentation=None):

        # Load image and mask
        print("image_id :  ", image_id)  # 打印载入图片的序号
        image = self.load_image(image_id)
        mask, class_ids = self.load_mask(image_id)
        original_shape = image.shape
        image, window, scale, padding = self.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)
        mask = self.resize_mask(mask, scale, padding)

        print('data_resize_image and resize_mask')

        # Random horizontal flips.
        # TODO: will be removed in a future update in favor of augmentation

        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        # Augmentation
        # This requires the imgaug lib (https://github.com/aleju/imgaug)
        if augmentation:
            import imgaug

            # Augmenters that are safe to apply to masks
            # Some, such as Affine, have settings that make them unsafe, so always
            # test your augmentation on masks
            MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                               "Fliplr", "Flipud", "CropAndPad",
                               "Affine", "PiecewiseAffine"]

            def hook(images, augmenter, parents, default):
                """Determines which augmenters to apply to masks."""
                return augmenter.__class__.__name__ in MASK_AUGMENTERS

            # Store shapes before augmentation to compare
            image_shape = image.shape
            mask_shape = mask.shape
            # Make augmenters deterministic to apply similarly to images and masks
            det = augmentation.to_deterministic()
            image = det.augment_image(image)
            # Change mask to np.uint8 because imgaug doesn't support np.bool
            mask = det.augment_image(mask.astype(np.uint8), hooks=imgaug.HooksImages(activator=hook))
            # Verify that shapes didn't change
            assert image.shape == image_shape, "Augmentation shouldn't change image size"
            assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
            # Change mask back to bool
            mask = mask.astype(np.bool)

        # Note that some boxes might be all zeros if the corresponding mask got cropped out.
        # and here is to filter them out
        _idx = np.sum(mask, axis=(0, 1)) > 0
        mask = mask[:, :, _idx]
        class_ids = class_ids[_idx]
        # Bounding boxes. Note that some boxes might be all zeros
        # if the corresponding mask got cropped out.
        # bbox: [num_instances, (y1, x1, y2, x2)]
        bbox = self.extract_bboxes(mask)

        # Active classes
        # Different datasets have different classes, so track the
        # classes supported in the dataset of this image.
        active_class_ids = np.ones([self.num_classes], dtype=np.int32)

        image_meta = np.array(
            [image_id] +  # size=1
            list(original_shape) +  # size=3
            list(image.shape) +  # size=3
            list(window) +  # size=4 (y1, x1, y2, x2) in image cooredinates
            [scale] +  # size=1
            list(active_class_ids)  # size=num_classes
        )

        print('using model data')
        return image, image_meta, class_ids, bbox, mask

    def compute_overlaps(self,boxes1, boxes2):
        # each value in boxes2 compute with all boxes1,and calling compute_iou function
        # finally, value save in [number_boxes1,number_boxes2]
        """Computes IoU overlaps between two sets of boxes.
        boxes1, boxes2: [N, (y1, x1, y2, x2)].

        For better performance, pass the largest set first and the smaller second.
        """
        # Areas of anchors and GT boxes
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
        # Each cell contains the IoU value.
        overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))  # building  variables for overlaps to save
        for i in range(overlaps.shape[1]):
            box2 = boxes2[i]
            y1 = np.maximum(box2[0], boxes1[:, 0])
            y2 = np.minimum(box2[2], boxes1[:, 2])
            x1 = np.maximum(box2[1], boxes1[:, 1])
            x2 = np.minimum(box2[3], boxes1[:, 3])
            intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
            union = area2[i] + area1[:] - intersection[:]
            overlaps[:, i] = intersection / union

        return overlaps

    def build_rpn_targets(self, anchors, gt_class_ids, gt_boxes, config):

        print('data_rpn_box')

        """Given the anchors and GT boxes, compute overlaps and identify positive
        anchors and deltas to refine them to match their corresponding GT boxes.

        anchors: [num_anchors, (y1, x1, y2, x2)]
        gt_class_ids: [num_gt_boxes] Integer class IDs.
        gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

        Returns:
        rpn_match: [N] (int32) matches between anchors and GT boxes.
                   1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        """
        # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
        # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
        rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances. Exclude
        # them from training. A crowd box is given a negative class ID.
        crowd_ix = np.where(gt_class_ids < 0)[0]
        if crowd_ix.shape[0] > 0:
            # Filter out crowds from ground truth class IDs and boxes
            non_crowd_ix = np.where(gt_class_ids > 0)[0]
            crowd_boxes = gt_boxes[crowd_ix]
            gt_class_ids = gt_class_ids[non_crowd_ix]
            gt_boxes = gt_boxes[non_crowd_ix]
            # Compute overlaps with crowd boxes [anchors, crowds]
            crowd_overlaps = self.compute_overlaps(anchors, crowd_boxes)
            crowd_iou_max = np.amax(crowd_overlaps, axis=1)
            no_crowd_bool = (crowd_iou_max < 0.001)
        else:
            # All anchors don't intersect a crowd
            no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

        # Compute overlaps [num_anchors, num_gt_boxes]
        overlaps = self.compute_overlaps(anchors, gt_boxes)

        # Match anchors to GT Boxes
        # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
        # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
        # Neutral anchors are those that don't match the conditions above,
        # and they don't influence the loss function.
        # However, don't keep any GT box unmatched (rare, but happens). Instead,
        # match it to the closest anchor (even if its max IoU is < 0.3).
        #
        # 1. Set negative anchors first. They get overwritten below if a GT box is
        # matched to them. Skip boxes in crowd areas.
        anchor_iou_argmax = np.argmax(overlaps, axis=1)
        anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
        rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
        # 2. Set an anchor for each GT box (regardless of IoU value).
        # If multiple anchors have the same IoU match all of them
        gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
        rpn_match[gt_iou_argmax] = 1
        # 3. Set anchors with high overlap as positive.
        rpn_match[anchor_iou_max >= 0.7] = 1

        # Subsample to balance positive and negative anchors
        # Don't let positives be more than half the anchors
        ids = np.where(rpn_match == 1)[0]
        extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
        if extra > 0:
            # Reset the extra ones to neutral
            ids = np.random.choice(ids, extra, replace=False)
            rpn_match[ids] = 0
        # Same for negative proposals
        ids = np.where(rpn_match == -1)[0]
        extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                            np.sum(rpn_match == 1))
        if extra > 0:
            # Rest the extra ones to neutral
            ids = np.random.choice(ids, extra, replace=False)
            rpn_match[ids] = 0

        # For positive anchors, compute shift and scale needed to transform them
        # to match the corresponding GT boxes.
        ids = np.where(rpn_match == 1)[0]
        ix = 0  # index into rpn_bbox
        # TODO: use box_refinement() rather than duplicating the code here
        for i, a in zip(ids, anchors[ids]):
            # Closest gt box (it might have IoU < 0.7)
            gt = gt_boxes[anchor_iou_argmax[i]]

            # Convert coordinates to center plus width/height.
            # GT Box
            gt_h = gt[2] - gt[0]
            gt_w = gt[3] - gt[1]
            gt_center_y = gt[0] + 0.5 * gt_h
            gt_center_x = gt[1] + 0.5 * gt_w
            # Anchor
            a_h = a[2] - a[0]
            a_w = a[3] - a[1]
            a_center_y = a[0] + 0.5 * a_h
            a_center_x = a[1] + 0.5 * a_w

            # Compute the bbox refinement that the RPN should predict.
            rpn_bbox[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w),
            ]
            # Normalize
            rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
            ix += 1

        return rpn_match, rpn_bbox

    def generate_random_rois(self, image_shape, count, gt_boxes):
        """Generates ROI proposals similar to what a region proposal network
        would generate.

        image_shape: [Height, Width, Depth]
        count: Number of ROIs to generate
        gt_class_ids: [N] Integer ground truth class IDs
        gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.

        Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
        """
        # placeholder
        rois = np.zeros((count, 4), dtype=np.int32)

        # Generate random ROIs around GT boxes (90% of count)
        rois_per_box = int(0.9 * count / gt_boxes.shape[0])
        for i in range(gt_boxes.shape[0]):
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
            h = gt_y2 - gt_y1
            w = gt_x2 - gt_x1
            # random boundaries
            r_y1 = max(gt_y1 - h, 0)
            r_y2 = min(gt_y2 + h, image_shape[0])
            r_x1 = max(gt_x1 - w, 0)
            r_x2 = min(gt_x2 + w, image_shape[1])

            # To avoid generating boxes with zero area, we generate double what
            # we need and filter out the extra. If we get fewer valid boxes
            # than we need, we loop and try again.
            while True:
                y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
                x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
                # Filter out zero area boxes
                threshold = 1
                y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                            threshold][:rois_per_box]
                x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                            threshold][:rois_per_box]
                if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                    break

            # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
            # into x1, y1, x2, y2 order
            x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
            y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
            box_rois = np.hstack([y1, x1, y2, x2])
            rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

        # Generate random ROIs anywhere in the image (10% of count)
        remaining_count = count - (rois_per_box * gt_boxes.shape[0])
        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
            x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                        threshold][:remaining_count]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                        threshold][:remaining_count]
            if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        global_rois = np.hstack([y1, x1, y2, x2])
        rois[-remaining_count:] = global_rois
        return rois

    def box_refinement(self,box, gt_box):
        """Compute refinement needed to transform box to gt_box.
        box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
        assumed to be outside the box.
        """
        box = box.astype(np.float32)
        gt_box = gt_box.astype(np.float32)

        height = box[:, 2] - box[:, 0]
        width = box[:, 3] - box[:, 1]
        center_y = box[:, 0] + 0.5 * height
        center_x = box[:, 1] + 0.5 * width

        gt_height = gt_box[:, 2] - gt_box[:, 0]
        gt_width = gt_box[:, 3] - gt_box[:, 1]
        gt_center_y = gt_box[:, 0] + 0.5 * gt_height
        gt_center_x = gt_box[:, 1] + 0.5 * gt_width

        dy = (gt_center_y - center_y) / height
        dx = (gt_center_x - center_x) / width
        dh = np.log(gt_height / height)
        dw = np.log(gt_width / width)

        return np.stack([dy, dx, dh, dw], axis=1)

    def build_detection_targets(self, rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
        """Generate targets for training Stage 2 classifier and mask heads.
        This is not used in normal training. It's useful for debugging or to train
        the Mask RCNN heads without using the RPN head.

        Inputs:
        rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
        gt_class_ids: [instance count] Integer class IDs
        gt_boxes: [instance count, (y1, x1, y2, x2)]
        gt_masks: [height, width, instance count] Ground truth masks. Can be full
                  size or mini-masks.

        Returns:
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
        bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
                bbox refinements.
        masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
               to bbox boundaries and resized to neural network output size.
        """
        assert rpn_rois.shape[0] > 0
        assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
            gt_class_ids.dtype)
        assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(
            gt_boxes.dtype)
        assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(
            gt_masks.dtype)

        # It's common to add GT Boxes to ROIs but we don't do that here because
        # according to XinLei Chen's paper, it doesn't help.

        # Trim empty padding in gt_boxes and gt_masks parts
        instance_ids = np.where(gt_class_ids > 0)[0]
        assert instance_ids.shape[0] > 0, "Image must contain instances."
        gt_class_ids = gt_class_ids[instance_ids]
        gt_boxes = gt_boxes[instance_ids]
        gt_masks = gt_masks[:, :, instance_ids]

        # Compute areas of ROIs and ground truth boxes.
        rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
                       (rpn_rois[:, 3] - rpn_rois[:, 1])
        gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
                      (gt_boxes[:, 3] - gt_boxes[:, 1])

        # Compute overlaps [rpn_rois, gt_boxes]
        overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
        for i in range(overlaps.shape[1]):
            gt = gt_boxes[i]
            overlaps[:, i] = self.compute_iou(
                gt, rpn_rois, gt_box_area[i], rpn_roi_area)

        # Assign ROIs to GT boxes
        rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
        rpn_roi_iou_max = overlaps[np.arange(
            overlaps.shape[0]), rpn_roi_iou_argmax]
        # GT box assigned to each ROI
        rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
        rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

        # Positive ROIs are those with >= 0.5 IoU with a GT box.
        fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

        # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
        # TODO: To hard example mine or not to hard example mine, that's the question
        # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
        bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

        # Subsample ROIs. Aim for 33% foreground.
        # FG
        fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
        if fg_ids.shape[0] > fg_roi_count:
            keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
        else:
            keep_fg_ids = fg_ids
        # BG
        remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
        if bg_ids.shape[0] > remaining:
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
        else:
            keep_bg_ids = bg_ids
        # Combine indices of ROIs to keep
        keep = np.concatenate([keep_fg_ids, keep_bg_ids])
        # Need more?
        remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
        if remaining > 0:
            # Looks like we don't have enough samples to maintain the desired
            # balance. Reduce requirements and fill in the rest. This is
            # likely different from the Mask RCNN paper.

            # There is a small chance we have neither fg nor bg samples.
            if keep.shape[0] == 0:
                # Pick bg regions with easier IoU threshold
                bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
                assert bg_ids.shape[0] >= remaining
                keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
                assert keep_bg_ids.shape[0] == remaining
                keep = np.concatenate([keep, keep_bg_ids])
            else:
                # Fill the rest with repeated bg rois.
                keep_extra_ids = np.random.choice(
                    keep_bg_ids, remaining, replace=True)
                keep = np.concatenate([keep, keep_extra_ids])
        assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
            "keep doesn't match ROI batch size {}, {}".format(
                keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

        # Reset the gt boxes assigned to BG ROIs.
        rpn_roi_gt_boxes[keep_bg_ids, :] = 0
        rpn_roi_gt_class_ids[keep_bg_ids] = 0

        # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
        rois = rpn_rois[keep]
        roi_gt_boxes = rpn_roi_gt_boxes[keep]
        roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
        roi_gt_assignment = rpn_roi_iou_argmax[keep]

        # Class-aware bbox deltas. [y, x, log(h), log(w)]
        bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                           config.NUM_CLASSES, 4), dtype=np.float32)
        pos_ids = np.where(roi_gt_class_ids > 0)[0]
        bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = self.box_refinement(
            rois[pos_ids], roi_gt_boxes[pos_ids, :4])
        # Normalize bbox refinements
        bboxes /= config.BBOX_STD_DEV

        # Generate class-specific target masks
        masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                         dtype=np.float32)
        for i in pos_ids:
            class_id = roi_gt_class_ids[i]
            assert class_id > 0, "class id must be greater than 0"
            gt_id = roi_gt_assignment[i]
            class_mask = gt_masks[:, :, gt_id]

            # if config.USE_MINI_MASK:
            #     # Create a mask placeholder, the size of the image
            #     placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            #     # GT box
            #     gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            #     gt_w = gt_x2 - gt_x1
            #     gt_h = gt_y2 - gt_y1
            #     # Resize mini mask to size of GT box
            #     placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
            #         np.round(utils.resize(class_mask, (gt_h, gt_w))).astype(bool)
            #     # Place the mini batch in the placeholder
            #     class_mask = placeholder

            # Pick part of the mask and resize it
            y1, x1, y2, x2 = rois[i].astype(np.int32)
            m = class_mask[y1:y2, x1:x2]
            mask = self.resize(m, config.MASK_SHAPE)
            masks[i, :, :, class_id] = mask

        return rois, roi_gt_class_ids, bboxes, masks

    def data_generator(self, config, shuffle=True, augment=False, augmentation=None,
                       random_rois=0, batch_size=1, detection_targets=False):

        b = 0  # batch item index
        image_index = -1
        image_ids = np.copy(self.image_ids)  # dataset.image_ids 运用了 @property

        error_count = 0

        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]

        backbone_shapes = \
            np.array([[int(math.ceil(config.IMAGE_SHAPE[0] / stride)),
                       int(math.ceil(config.IMAGE_SHAPE[1] / stride))] for stride in
                      config.BACKBONE_STRIDES])  # BACKBONE_STRIDES = [4, 8, 16, 32, 64]

        # compute_backbone_shapes(config, config.IMAGE_SHAPE)  # (5,2) # [4, 8, 16, 32, 64]

        anchors = self.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,  # (8, 16, 32, 64, 128)
                                           config.RPN_ANCHOR_RATIOS,  # [0.5, 1, 2]
                                           backbone_shapes,  # image_shape / [4, 8, 16, 32, 64] is five rows 2 cols
                                           config.BACKBONE_STRIDES,  # [4, 8, 16, 32, 64]
                                           config.RPN_ANCHOR_STRIDE)  # =1

        print('data_class_data_anchors')

        # 【n,4】
        # 得到的anchor数量为 每个scale分别是3*(image_shape/4)**2,3*(image_shape/8)**2,3*(image_shape/16)**2,
        # 3*(image_shape/4)**2,3*(image_shape/64)**2,

        # Keras requires a generator to run indefinitely.
        while True:
            try:
                # Increment index to pick next image. Shuffle if at the start of an epoch.
                image_index = (image_index + 1) % len(image_ids)
                if shuffle and image_index == 0:
                    np.random.shuffle(image_ids)

                # Get GT bounding boxes and masks for image.
                image_id = image_ids[image_index]

                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    self.load_image_gt(config, image_id, augment=augment,
                                  augmentation=augmentation)

                # Skip images that have no instances. This can happen in cases
                # where we train on a subset of classes and the image doesn't
                # have any of the classes we care about.
                if not np.any(gt_class_ids > 0):
                    continue

                # RPN Targets
                rpn_match, rpn_bbox = self.build_rpn_targets(anchors, gt_class_ids, gt_boxes, config)

                # Mask R-CNN Targets
                if random_rois:
                    rpn_rois = self.generate_random_rois(image.shape, random_rois, gt_boxes)
                    if detection_targets:
                        rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = \
                            self.build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)

                # Init batch arrays
                if b == 0:
                    batch_image_meta = np.zeros((batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                    batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                    batch_rpn_bbox = np.zeros([batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                    batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                    batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                    batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                    batch_gt_masks = np.zeros(
                        (batch_size, gt_masks.shape[0], gt_masks.shape[1], config.MAX_GT_INSTANCES),
                        dtype=gt_masks.dtype)
                    if random_rois:
                        batch_rpn_rois = np.zeros((batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                        if detection_targets:
                            batch_rois = np.zeros((batch_size,) + rois.shape, dtype=rois.dtype)
                            batch_mrcnn_class_ids = np.zeros((batch_size,) + mrcnn_class_ids.shape,
                                                             dtype=mrcnn_class_ids.dtype)
                            batch_mrcnn_bbox = np.zeros((batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                            batch_mrcnn_mask = np.zeros((batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

                # If more instances than fits in the array, sub-sample from them.
                if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                    ids = np.random.choice(np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                    gt_class_ids = gt_class_ids[ids]
                    gt_boxes = gt_boxes[ids]
                    gt_masks = gt_masks[:, :, ids]

                # Add to batch
                batch_image_meta[b] = image_meta
                batch_rpn_match[b] = rpn_match[:, np.newaxis]
                batch_rpn_bbox[b] = rpn_bbox
                batch_images[b] = image.astype(np.float32) - config.MEAN_PIXEL
                batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
                batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
                batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
                if random_rois:
                    batch_rpn_rois[b] = rpn_rois
                    if detection_targets:
                        batch_rois[b] = rois
                        batch_mrcnn_class_ids[b] = mrcnn_class_ids
                        batch_mrcnn_bbox[b] = mrcnn_bbox
                        batch_mrcnn_mask[b] = mrcnn_mask
                b += 1

                # Batch full?
                if b >= batch_size:
                    inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                              batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                    outputs = []

                    if random_rois:
                        inputs.extend([batch_rpn_rois])
                        if detection_targets:
                            inputs.extend([batch_rois])
                            # Keras requires that output and targets have the same number of dimensions
                            batch_mrcnn_class_ids = np.expand_dims(
                                batch_mrcnn_class_ids, -1)
                            outputs.extend(
                                [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])
                    print('data_load_finish')
                    yield inputs , outputs
                    # start a new batch
                    b = 0
            except:
                raise Exception("not pass")
            '''
            可能会抛出异常，属于正常，因为出现生成器销毁而出现的。
            Exception ignored in: <generator object Dataset_data.data_generator at 0x000002002D40BB48>
            Traceback (most recent call last):
            File "C:/Users/51102/Desktop/MASKRCNN_tangjun/Mask_RCNN-master/train_demo.py", line 1249, in data_generator
            raise Exception("not pass")
            Exception: not pass
            '''





def train_model():
    img_floder ='C:\\Users\\51102\\Desktop\\maskrcnn(tangjun)\\1021'  ####################################################################################################
    dataset_train = Dataset_data()
    dataset_train.data_load_information(img_floder)
    model = modellib.MaskRCNN(mode="training", config=config)

    COCO_MODEL_PATH='C:\\Users\\51102\\Desktop\\maskrcnn(tangjun)\\mask_rcnn_shapes_0002.h5'
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])

    # 产生数据
    train_generator = dataset_train.data_generator(config, shuffle=True,
                                                   augmentation=None,
                                                   batch_size=config.batch_size)

    model.train(train_generator,
                learning_rate=config.LEARNING_RATE,
                epochs=4,
                layers='heads')




    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    # model.train(dataset_train, dataset_train,
    #             learning_rate=config.LEARNING_RATE / 10,
    #             epochs=3,
    #             layers="all")







from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon
import colorsys

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        # cv.rectangle(masked_image, (y1[0],x1[0]), (y2[0],x2[0]), (0, 250, 0), 2)     # 自己添加代码
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
    return masked_image










def predict():
    import skimage.io

    # Create models in training mode
    config = Predict_Config()
    config.display()
    model = modellib.MaskRCNN(mode="inference", config=config)

    # model_path = 'C:\\Users\\51102\\Desktop\mask-rcnn-me\\MASKRCNN_myself\Mask_RCNN-master\\logs\\shapes20200216T1602\\mask_rcnn_shapes_0002.h5'
    model_path = 'C:\\Users\\51102\\Desktop\\maskrcnn(tangjun)\\log\\04.h5'

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    class_names = ['BG', 'line_bulge','dot_concave','dot_bulge','Irregular_concave']

    # file_names ='C:\\Users\\51102\\Desktop\\maskrcnn(tangjun)\\1.jpg'

    file_names='C:\\Users\\51102\\Desktop\\maskrcnn(tangjun)\\3.bmp'

    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    image = skimage.io.imread(file_names)
    image=image[:, :, 0:3]
    print('image=', image.shape)

    # Run detection
    results = model.detect([image], log_print=1)
    '''
                results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks})
    '''

    # Visualize results
    r = results[0]
    print('r=',r)
    display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])






if __name__ == "__main__":
    train_model()
    # predict()





