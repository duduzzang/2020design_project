from pathlib import Path
import cv2
import numpy as np


def line_scan(img):
    h, w = img.shape[:2]

    max_coord = []
    max_length = 0
    for line_idx, line in enumerate(img):
        min_x = 0
        max_x = w - 1

        for idx in range(w):
            if line[idx] == 0:
                min_x += 1
            else:
                break
        for idx in reversed(range(w)):
            if line[idx] == 0:
                max_x -= 1
            else:
                break

        line_length = max_x - min_x + 1
        if line_length > max_length:
            max_length = line_length

            max_coord = [line_idx, min_x, max_x]

    return max_coord


if __name__ == "__main__":
    input_root = Path("C:/Users/hansol/PycharmProjects/unet++/pytorch-nested-unet/outputs/Prostate_dataset_512_NestedUNet_woDS_selected/0177/0")

    image_name = "A_0177_03.jpg"
    save_name = "A_0177_03_length2.jpg"

    img_np = cv2.imread(str(input_root / image_name))
    img_np_binary = cv2.imread(str(input_root / image_name), 0)
    img_np_binary = (img_np_binary > 127).astype(np.uint8) * 255

    max_coord = line_scan(img_np_binary)

    start = (max_coord[1], max_coord[0])
    end = (max_coord[2], max_coord[0])

    img_out = cv2.line(img_np, start, end, (0, 0, 255), 2)

    cv2.imwrite(str(input_root / save_name), img_out)
    # cv2.imshow("line", img_out)
    # cv2.waitKey(0)
