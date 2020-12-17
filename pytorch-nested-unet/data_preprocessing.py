from glob import glob
import os
import json
import pydicom
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
## preprocessing
if __name__ == '__main__':
    path = 'C:\\Users\\hansol\\Downloads\\Prostate US AI _0063\\'

    img_size_r = 512
    img_size_c = 384

    int_num = 0
    int_Ulidx = 0
    int_niidx = 0
    a = glob(path+'*')
    for pname in tqdm(a):
        patient_name = pname[-4:]
        int_num = int_num + 1
        UIDlist=[]
        niilist=[]
        jsnlist=[]

        with open(pname+'\\lesionAnnot2D.json', 'r') as fname:
            jdata = json.load(fname)


        for mask in jdata['lesionAnnot']:
            a = mask
            b = a['imageInfo']
            b_anno = a['userComment']
            c = b['imageSOPInstUID']
            UIDlist.append(c)
            d = a['maskFileNameNifti']
            niilist.append(d)
            f = b_anno['annotation']
            jsnlist.append(f)

        DicomPath = glob(pname + '\\DICOM\\*')
        MaskPath = glob(pname + '\\MASK\\*')
        # print(DicomPath)
        Dicomdict = {}
        Maskdict = {}

        for dicomidx in DicomPath:
            fuck = pydicom.dcmread(dicomidx)
            arr = fuck.SOPInstanceUID
            arr2 = fuck.pixel_array
            if arr2.ndim == 3:
                arr2 = arr2[:,:,0]
            elif arr2.ndim == 4:
                arr2 = arr2[0,:,:,0]
            Dicomdict[arr] = arr2

        for maskidx in MaskPath:
            proxy = nib.load(maskidx)
            suck = proxy.get_fdata()
            suck2 = suck.reshape(suck.shape[0], suck.shape[1])
            suck2 = np.transpose(suck2)

            Maskdict[maskidx[-24:]] = suck2

        inputimg = []
        inputmask = []
        for Ulidx in range(len(UIDlist)):
            if UIDlist[Ulidx] in Dicomdict:
                dcm_arr = Dicomdict.get(UIDlist[Ulidx])
                resize_img = cv2.resize(dcm_arr, (img_size_r, img_size_c))
                imgpath = 'C:\\Users\\hansol\\PycharmProjects\\unet++\\pytorch-nested-unet\\inputs\\Prostate_dataset_512\\images\\'
                cv2.imwrite(os.path.join(imgpath, '{}_{}_{}.png') .format(jsnlist[Ulidx], patient_name, str(Ulidx).zfill(2)), resize_img)

        for niidx in range(len(niilist)):
            if niilist[niidx] in Maskdict:
                mask_arr = Maskdict.get(niilist[niidx])
                resize_mask = cv2.resize(mask_arr, (img_size_r, img_size_c))
                imgpath = 'C:\\Users\\hansol\\PycharmProjects\\unet++\\pytorch-nested-unet\\inputs\\Prostate_dataset_512\\masks\\0\\'
                cv2.imwrite(os.path.join(imgpath, '{}_{}_{}.png') .format(jsnlist[niidx], patient_name, str(niidx).zfill(2)), resize_mask)

