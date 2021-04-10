import os
import time
from flickrapi import FlickrAPI
import requests
import sys

KEY = 'c7f6e255cd1b9480ce8c4597dfc2b8af'
SECRET = '20c6f89f52aa873a'

tags = [
    'flame'
]
images_per_tag = 1000

def download():
    for tag in tags:

        print('Getting urls for', tag)
        urls = get_urls(tag,images_per_tag)

        print('Downlaing images for', tag)
        path = os.path.join('data', tag)

        download_images(urls, path)


# List of sizes:
# url_o: Original (4520 × 3229)
# url_k: Large 2048 (2048 × 1463)
# url_h: Large 1600 (1600 × 1143)
# url_l=: Large 1024 (1024 × 732)
# url_c: Medium 800 (800 × 572)
# url_z: Medium 640 (640 × 457)
# url_m: Medium 500 (500 × 357)
# url_n: Small 320 (320 × 229)
# url_s: Small 240 (240 × 171)
# url_t: Thumbnail (100 × 71)
# url_q: Square 150 (150 × 150)
# url_sq: Square 75 (75 × 75)

SIZES = ["url_m", "url_n", "url_o", "url_k", "url_h", "url_l", "url_c"]  # order of preference

def get_photos(image_tag):
    extras = ','.join(SIZES)
    flickr = FlickrAPI(KEY, SECRET)
    photos = flickr.walk(text=image_tag,
                            extras=extras,  # get the url for the original size image
                            privacy_filter=1,  # search only for public photos
                            content_type=1, # only photos
                            media='photos',
                            per_page=500,
                            sort='interestingness-desc')
    return photos

def get_url(photo):
    for i in range(len(SIZES)):
        url = photo.get(SIZES[i])
        if url:  # if url is None try with the next size
            return url

def get_urls(image_tag, max):
    photos = get_photos(image_tag)
    counter=0
    urls=[]

    for photo in photos:
        if counter < max:
            url = get_url(photo)  # get preffered size url
            if url:
                urls.append(url)
                counter += 1
            # if no url for the desired sizes then try with the next photo
        else:
            break

    return urls


def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def download_images(urls, path):
    create_folder(path)  # makes sure path exists

    for url in urls:
        image_name = url.split("/")[-1]
        image_path = os.path.join(path, image_name)

        if not os.path.isfile(image_path):  # ignore if already downloaded
            response=requests.get(url,stream=True)

            with open(image_path,'wb') as outfile:
                outfile.write(response.content)

start_time = time.time()

download()

print('Took', round(time.time() - start_time, 2), 'seconds')
