import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))

def read_image_from_path(path, size):
    img = Image.open(path).convert('RGB').resize(size)
    return np.array(img)

def folder_to_images(folder, size):
    list_dir = [folder + '/' + name for name in os.listdir(folder)]
    # path cua cac anh 
    images_np = np.zeros(shape = (len(list_dir), *size, 3))
    images_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_image_from_path(path, size)
        images_path.append(path)
    images_path = np.array(images_path)
    return images_np, images_path

def absolute_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum((np.abs(query - data)), axis = axis_batch_size)

def cosine_similarity(query, data):
    axis_bath_size = tuple(range(1, len(data.shape)))
    query_norm = np.sqrt(np.sum(query ** 2))
    data_norm = np.sqrt(np.sum(data ** 2, axis = axis_bath_size))
    return np.sum(query * data, axis = axis_bath_size) / (query_norm * data_norm + np.finfo(np.float32).eps)

def get_l1_scores(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    ls_path_scores = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size)
            rates = absolute_difference(query, images_np)
            ls_path_scores.extend(zip(images_path, rates))
    return query, ls_path_scores      

def get_cosine_scores(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    ls_cosine_scores = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size)
            rates = cosine_similarity(query, images_np)
            ls_cosine_scores.extend(zip(images_path, rates))
    return query, ls_cosine_scores

def plot_results(querquery_pathy, ls_path_score, reverse):
    fig = plt.figure(figsize=(15, 9))
    fig.add_subplot(2, 3, 1)
    plt.imshow(read_image_from_path(querquery_pathy, size=(448,448)))
    plt.title(f"Query Image: {querquery_pathy.split('/')[2]}", fontsize=16)
    plt.axis("off")
    for i, path in enumerate(sorted(ls_path_score, key=lambda x : x[1], reverse=reverse)[:5], 2):
        fig.add_subplot(2, 3, i)
        plt.imshow(read_image_from_path(path[0], size=(448,448)))
        plt.title(f"Top {i-1}: {path[0].split('/')[2]}", fontsize=16)
        plt.axis("off")
    plt.show()

query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
root_img_path = f"{ROOT}/train/"
size = (400, 400)
query, ls_path_scores = get_cosine_scores(root_img_path, query_path, size)
plot_results(query_path, ls_path_scores, reverse=True)

