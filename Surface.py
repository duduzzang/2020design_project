import cv2
import shutil
import numpy as np

from pathlib import Path
from collections import defaultdict


if __name__ == "__main__":
    file_root = Path("C:/Users/hansol/PycharmProjects/unet++/pytorch-nested-unet/outputs/Prostate_dataset_512_NestedUNet_woDS")

    image_list = [name for name in file_root.glob("**/*.jpg")]
    image_list.sort()

    surface_dict_list = [defaultdict(lambda: [[], []]), defaultdict(lambda: [[], []])]

    for image_name in image_list:
        folder_number = int(list(image_name.parents)[0].name)
        surface_dict = surface_dict_list[folder_number]

        mask_image = cv2.imread(str(image_name))
        mask_image = mask_image[:, :, 0]
        index_mask_img = np.where(mask_image >= 127.5)
        patientnum = str(image_name)[-11:-7]
        surface_dict[patientnum][0].append(len(index_mask_img[0]))
        surface_dict[patientnum][1].append(str(image_name))

    output_root = Path("C:/Users/hansol/PycharmProjects/unet++/pytorch-nested-unet/outputs/Prostate_dataset_512_NestedUNet_woDS_selected")
    for surface_dict in surface_dict_list:
        for k, v in surface_dict.items():
            (output_root / k / "0").mkdir(parents=True, exist_ok=True)
            (output_root / k / "1").mkdir(parents=True, exist_ok=True)

            pixel_list = np.array(v[0])
            max_index = int(np.argmax(pixel_list))

            input_image_name = v[1][max_index]
            image_name = Path(input_image_name).name
            shutil.copy(input_image_name, output_root / k / f"{'0' if image_name[0] == 'A' else '1'}" / image_name)
