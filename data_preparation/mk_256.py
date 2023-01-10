import os
import numpy as np
from einops import rearrange
import cv2

file_path = '../levir_cd/'
train_types = ['train', 'test', 'val']
file_numbers = ['A', 'B', 'label']
write_paths = '../levir_cd_256/'


def read_file(file_name, write_pathss):
    print(file_name)
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    img = rearrange(img, '(b1 h) (b2 w) c -> (b1 b2) h w c', b1=4, b2=4)
    write_name = file_name.split('/')[-1].split('.')[0]
    for i in range(img.shape[0]):
        cv2.imwrite(os.path.join(write_pathss, write_name + f'_{i}.png'), img[i])


for train_type in train_types:
    for file_number in file_numbers:
        file_address = os.path.join(file_path, train_type, file_number)
        write_path = os.path.join(write_paths, train_type, file_number)
        file_list = sorted(os.listdir(file_address))
        for i in file_list:
            file_ls = os.path.join(file_address, i)
            read_file(file_ls, write_path)




