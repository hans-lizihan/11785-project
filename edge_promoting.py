import cv2
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, process


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def edge_promoting_runner(list_of_files, thread_index, root, save):
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)
    n = 1

    for f in tqdm(list_of_files, position=thread_index):
        rgb_img = cv2.imread(os.path.join(root, f))
        gray_img = cv2.imread(os.path.join(root, f), 0)
        rgb_img = cv2.resize(rgb_img, (256, 256))
        pad_img = np.pad(rgb_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
        gray_img = cv2.resize(gray_img, (256, 256))
        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(rgb_img)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(
                pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(
                pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(
                pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

        # result = np.concatenate((rgb_img, gauss_img), 1)
        result = gauss_img

        cv2.imwrite(os.path.join(
            save, f.split('/')[-1]), gauss_img)

        n += 1


def edge_promoting(root, save):
    file_list = os.listdir(root)
    num_cores = os.cpu_count()
    if not os.path.isdir(save):
        os.makedirs(save)

    split_file_list = split(file_list, num_cores)

    processes = []
    for i, s in enumerate(split_file_list):
        p = Process(
            target=edge_promoting_runner,
            args=(s, i, root, save)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == '__main__':
    edge_promoting('dataset/TgtDataSet/Kimetsu no Yaiba/', 'dataset/TgtDataSet/Kimetsu no Yaiba_smooth/')
