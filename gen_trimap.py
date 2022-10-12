import os
import cv2
import numpy as np

folder = os.listdir('./sil/00009')
for j in range(len(folder)):
    path = './sil/00009/'+folder[j]
    output_path = './trimap/00009/'+folder[j]
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    name = os.listdir(path)

    for i in range(len(name)):
        alpha = cv2.imread(os.path.join(path, name[i]))/255.
        fg_kernel = np.ones((8, 8), np.uint8)
        bg_kernel = np.ones((12, 12), np.uint8)
    
        fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, fg_kernel)
        bg_mask = cv2.erode(bg_mask, bg_kernel)

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        cv2.imwrite(os.path.join(output_path, name[i]), trimap)

