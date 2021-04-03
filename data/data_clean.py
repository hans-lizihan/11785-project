import os
import glob
from PIL import Image, ImageStat
from tqdm import tqdm

def is_grayscale(path="image.jpg"):
    im = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(im)
    if sum(stat.sum)/3 == stat.sum[0]: #check the avg with any element value
        return True #if grayscale
    else:
        return False #else its colour

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

files = glob.glob('./data/src_data/**/*.jpg')

def remove_gray_scale_images():
    for file in tqdm(files):
        if is_grayscale(file):
            os.remove(file)

def center_crop():
    for file in tqdm(files):
        im_new = crop_max_square(Image.open(file))
        im_new.save(file, quality=100)

def crop_256():
    for file in tqdm(files):
        im_new = Image.open(file).resize((256, 256))
        im_new.save(file)

crop_256()
