import os
import numpy as np
from src.utils.data_loader import read_image_from_path, folder_to_images

def absolute_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum((np.abs(query - data)), axis = axis_batch_size)

def cosine_similarity(query, data):
    axis_bath_size = tuple(range(1, len(data.shape)))
    query_norm = np.sqrt(np.sum(query ** 2))
    data_norm = np.sqrt(np.sum(data ** 2, axis = axis_bath_size))
    return np.sum(query * data, axis = axis_bath_size) / (query_norm * data_norm + np.finfo(np.float32).eps)

def get_l1_scores(root_img_path, query_path, size, class_names):
    query = read_image_from_path(query_path, size)
    ls_path_scores = []
    for folder in os.listdir(root_img_path):
        if folder in class_names:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size)
            rates = absolute_difference(query, images_np)
            ls_path_scores.extend(zip(images_path, rates))
    return query, ls_path_scores      

def get_cosine_scores(root_img_path, query_path, size, class_names):
    query = read_image_from_path(query_path, size)
    ls_cosine_scores = []
    for folder in os.listdir(root_img_path):
        if folder in class_names:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size)
            rates = cosine_similarity(query, images_np)
            ls_cosine_scores.extend(zip(images_path, rates))
    return query, ls_cosine_scores
