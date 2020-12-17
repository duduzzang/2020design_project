import cv2
import numpy as np
from pathlib import Path

import math

from line_scan import line_scan


if __name__ == "__main__":
    input_root = Path("C:/Users/hansol/PycharmProjects/unet++/pytorch-nested-unet/outputs/Prostate_dataset_512_NestedUNet_woDS_selected/0177/1")

    image_name = "S_0177_07.jpg"
    save_name = "S_0177_07_length5.jpg"

    img_np = cv2.imread(str(input_root / image_name))
    img_np_binary = cv2.imread(str(input_root / image_name), 0)
    img_np_binary = (img_np_binary > 127).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(img_np_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #### 제일 큰 contour 찾는걸로 바꿔야돼!
    contours = contours[0]
    x_coords = contours[:, :, 0]
    y_coords = contours[:, :, 1]

    length = (x_coords - np.transpose(x_coords, (1, 0))) ** 2 + (y_coords - np.transpose(y_coords, (1, 0))) ** 2

    max_index = np.argmax(length)

    i = max_index // length.shape[0]
    j = max_index % length.shape[0]

    i_coord = (int(contours[i, :, 0]), int(contours[i, :, 1]))
    j_coord = (int(contours[j, :, 0]), int(contours[j, :, 1]))

    output_img = cv2.line(img_np, i_coord, j_coord, (0, 0, 255), 2)

    theta = math.atan((j_coord[1] - i_coord[1]) / (j_coord[0] - i_coord[0]))

    h, w = img_np.shape[:2]

    affine_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), 90 + 180 / 3.141592 * theta, 1)
    rotate_img = cv2.warpAffine(img_np_binary, affine_matrix, (w, h))

    max_coord = line_scan(rotate_img)

    start = [max_coord[1], max_coord[0], 1]
    end = [max_coord[2], max_coord[0], 1]

    affine_matrix = np.concatenate([affine_matrix, [[0, 0, 1]]], axis=0)

    origin_start = (np.linalg.inv(affine_matrix) @ start)[:-1]
    origin_end = (np.linalg.inv(affine_matrix) @ end)[:-1]
    #
    output_img = cv2.line(output_img,
                          (int(origin_start[0]), int(origin_start[1])),
                          (int(origin_end[0]), int(origin_end[1])),
                          (0, 0, 255), 2)

    cv2.imwrite(str(input_root / save_name), output_img)
    # cv2.imshow("line", output_img)
    # cv2.waitKey(0)