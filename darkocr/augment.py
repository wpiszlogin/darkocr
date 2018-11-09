import random

from PIL import Image, ImageFilter
import Augmentor
from Augmentor.Operations import Operation
import numpy as np

augment_path = 'train_data/aug/'


# below classes extends Augmentor
# erosion makes thinner line, dilation makes it thicker
class ErosImage(Operation):
    def __init__(self, probability, max_erosion, max_dilation, pixels_mean=None):
        Operation.__init__(self, probability)
        self.max_erosion = max_erosion
        self.max_dilation = max_dilation
        self.pixels_mean = pixels_mean

    def perform_operation(self, image):
        image_mod = []
        for im in image:
            # r choose way and scale of change
            r = random.randint(-self.max_erosion, self.max_dilation)

            # use pixels mean of character
            if self.pixels_mean is not None:
                # calc mean of this image
                im_array = np.array(im)
                im_pixels = np.sum(im_array)
                thin_factor = (self.pixels_mean - im_pixels) / self.pixels_mean
                r = r + int(3 * thin_factor)

            # change to odd number
            r_odd = abs(2 * r - 1)
            im_mod = im
            if r > 0:
                im_mod = im.filter(ImageFilter.MaxFilter(r_odd))
            elif r < 0:
                im_mod = im.filter(ImageFilter.MinFilter(r_odd))
            image_mod.append(im_mod)

        return image_mod


# makes smooth edge, removes spikes
class SmoothImage(Operation):
    def __init__(self, probability, blur=1, threshold=128):
        Operation.__init__(self, probability)
        self.blur = blur
        self.threshold = threshold

    def perform_operation(self, image):
        image_mod = []
        for im in image:
            # blur image
            imbl = im.filter(ImageFilter.GaussianBlur(radius=self.blur))
            # set color by a threshold
            im_mod = imbl.point(lambda x: 0 if x < self.threshold else 255)
            image_mod.append(im_mod)

        return image_mod


# makes variations of aspect ratio in a smart way
class AutoresizeImage(Operation):
    def __init__(self, probability, max_horizontal_loss, max_vertical_loss):
        Operation.__init__(self, probability)
        self.max_horizontal_loss = max_horizontal_loss
        self.max_vertical_loss = max_vertical_loss

    def perform_operation(self, image):
        image_mod = []
        for im in image:
            # if no modifications
            im_mod = im
            # calc bounding box
            bbox = im.getbbox()
            margin_tol = 0.15
            horiz_margin_tol = margin_tol * im.width
            vert_margin_tol = margin_tol * im.height
            # modify if object covers whole picture
            if bbox[0] < horiz_margin_tol and bbox[1] < vert_margin_tol and im.width - bbox[2] < horiz_margin_tol and im.height - bbox[3] < vert_margin_tol:
                # here image will be paste
                im_mod = Image.new('L', (im.width, im.height))
                if random.randint(0, 10) <= 7:
                    # resize horizontally
                    scal_factor = (1 - self.max_horizontal_loss * random.random())
                    i2 = im.resize((int(scal_factor*im.width), im.height))
                else:
                    # resize vertically
                    scal_factor = (1 - self.max_vertical_loss * random.random())
                    i2 = im.resize((im.width, int(scal_factor*im.height)))

                im_mod.paste(i2, (int(im.width/2 - i2.width/2), int(im.height/2 - i2.height/2)))

            image_mod.append(im_mod)

        return image_mod


# adds margins around an object
class ExpandImage(Operation):
    def __init__(self, probability, width, height):
        Operation.__init__(self, probability)
        self.width = width
        self.height = height

    def perform_operation(self, image):
        image_mod = []
        for im in image:
            # if no modifications
            im_mod = im
            # if the image is smaller than a destination dimension
            if self.width > im.width and self.height > im.height:
                im_mod = Image.new('L', (self.width, self.height))
                im_mod.paste(im, (int(self.width/2 - im.width/2), int(self.height/2 - im.height/2)))

            image_mod.append(im_mod)

        return image_mod


# finds an object on a picture and crop it, standardizes margins
class AutoCropImage(Operation):
    def __init__(self, probability, margin):
        Operation.__init__(self, probability)
        self.margin = margin

    def perform_operation(self, image):
        image_mod = []
        for im in image:
            # calc bounding box
            bbox = im.getbbox()
            # find bigger dimension
            max_dim = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            min_dim = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
            margin = int(self.margin * max_dim)
            diff = int((max_dim - min_dim) / 2)

            crop_box = (
                max(0, bbox[0] - margin),
                max(0, bbox[1] - margin),
                min(im.width, bbox[2] + margin),
                min(im.height, bbox[3] + margin)
            )
            if diff > 0:
                if bbox[2] - bbox[0] > bbox[3] - bbox[1]:
                    crop_box = (
                        max(0, bbox[0] - margin),
                        max(0, bbox[1] - margin - diff),
                        min(im.width, bbox[2] + margin),
                        min(im.height, bbox[3] + margin + diff)
                    )
                else:
                    crop_box = (
                        max(0, bbox[0] - margin - diff),
                        max(0, bbox[1] - margin),
                        min(im.width, bbox[2] + margin + diff),
                        min(im.height, bbox[3] + margin)
                    )
            im_mod = im.crop(crop_box)
            image_mod.append(im_mod)

        return image_mod


# process folder containing png images
def augment_folder(path=augment_path, generated_count=50, pixels_mean=None):
    # temporary margin to make sure the object won't oversize during deformation
    temp_margin = 2.0
    # margin after operations = final_margin * size
    final_margin = 0.05
    # temporary increases size to make erosion softer
    erosion_scaling = 4

    # detect images
    p = Augmentor.Pipeline(path, output_directory="augment")
    # get dimension of first image
    im_width = 100
    im_height = 100
    if len(p.augmentor_images) > 0:
        first_path = p.augmentor_images[0].image_path
        im = Image.open(first_path)
        im_width = im.width
        im_height = im.height

    p.random_distortion(1, 2, 2, 8)
    p.add_operation(AutoresizeImage(probability=1, max_horizontal_loss=0.4, max_vertical_loss=0.2))
    p.add_operation(ExpandImage(probability=1, width=int(temp_margin*im_width), height=int(temp_margin*im_height)))
    p.rotate(probability=1, max_left_rotation=7, max_right_rotation=7)
    p.skew(1, 0.2)
    # p.shear(1, 5, 5)
    p.add_operation(AutoCropImage(probability=1, margin=final_margin))
    p.resize(1, int(erosion_scaling * im_width), int(erosion_scaling * im_height))
    p.add_operation(SmoothImage(probability=1, blur=2, threshold=128))
    pixels_mean_scaled = None

    if pixels_mean is not None:
        pixels_mean_scaled = pixels_mean * erosion_scaling * erosion_scaling * 255  # max of pickle data pixel = 1

    p.add_operation(ErosImage(probability=1, max_erosion=3, max_dilation=4, pixels_mean=pixels_mean_scaled))
    p.resize(1, int(im_width), int(im_height))
    p.sample(generated_count)
