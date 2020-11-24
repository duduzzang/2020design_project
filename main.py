from glob import glob
import os
import json
import pydicom
import pylibjpeg
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import time
from tqdm import tqdm
from PIL import Image

if __name__ == '__main__':
    path = 'C:\\Users\\hansol\\Downloads\\Prostate US AI _0063\\'
    img_size = 224
    int_num = 0
    int_Ulidx = 0
    int_niidx = 0
    a = glob(path+'*')
    for pname in tqdm(a): ##다음 환자
        # print(pname)
        int_num = int_num + 1
        ##json 접근 & list 생성##
        UIDlist=[]
        niilist=[]

        with open(pname+'\\lesionAnnot2D.json', 'r') as fname:
            jdata = json.load(fname)


        for mask in jdata['lesionAnnot']:
            a = mask
            b = a['imageInfo']
            c = b['imageSOPInstUID']
            UIDlist.append(c)
            d = a['maskFileNameNifti']
            niilist.append(d)
        # print(UIDlist)
        # print(niilist)


        DicomPath = glob(pname+'\\DICOM\\*') ##Dicom폴더 접근
        MaskPath = glob(pname + '\\MASK\\*') ##Mask폴더 접근
        # print(DicomPath)
        Dicomdict = {}
        Maskdict = {}

        for dicomidx in DicomPath:
            # print(dicomidx)
            fuck = pydicom.dcmread(dicomidx)
            arr = fuck.SOPInstanceUID
            arr2 = fuck.pixel_array
            if arr2.ndim == 3:
                arr2 = arr2[:,:,0] #### 채널수 정하기 ####
            elif arr2.ndim == 4:
                arr2 = arr2[0,:,:,0]
            Dicomdict[arr] = arr2

        for maskidx in MaskPath:
            proxy = nib.load(maskidx)
            suck = proxy.get_fdata()
            suck2 = suck.reshape(suck.shape[0], suck.shape[1])
            suck2 = np.transpose(suck2)

            Maskdict[maskidx[-24:]] = suck2

        # for ddkey in Dicomdict:
        #     print(ddkey)
        inputimg = []
        inputmask = []
        for Ulidx in range(len(UIDlist)):
            if UIDlist[Ulidx] in Dicomdict:
                int_Ulidx = int_Ulidx + 1
                dcm_name = int(int_Ulidx + int_num)
                dcm_arr = Dicomdict.get(UIDlist[Ulidx])
                # print(UIDlist[Ulidx])
                resize_img = cv2.resize(dcm_arr, (img_size, img_size))
                imgpath = 'C:\\Users\\hansol\\PycharmProjects\\unet++\\pytorch-nested-unet\\inputs\\dsb2018_224\\images\\'
                cv2.imwrite(os.path.join(imgpath, '%d.png') % dcm_name, resize_img)
                # cv2.imshow('gray', resize_img)
                # cv2.waitKey(60)

                # img_dcm = Image.fromarray(dcm_arr)
                # img_dcm = Image.fromarray(dcm_arr.astype('uint8'), 'L')
                # plt.imsave('%d.png' % dcm_name, img_dcm) ###개수?세는거..
                # plt.imshow(img_dcm, 'gray')
                # plt.show()

        for niidx in range(len(niilist)):
            if niilist[niidx] in Maskdict:
                int_niidx = int_niidx + 1
                mask_name = int(int_niidx + int_num)
                mask_arr = Maskdict.get(niilist[niidx])
                # print(niilist[niidx])
                resize_mask = cv2.resize(mask_arr, (img_size, img_size))
                imgpath = 'C:\\Users\\hansol\\PycharmProjects\\unet++\\pytorch-nested-unet\\inputs\\dsb2018_224\\masks\\0\\'
                cv2.imwrite(os.path.join(imgpath, '%d.png') % mask_name, resize_mask)
        #
        #         cv2.imshow('gray', mask_arr)
        #         cv2.waitKey(60)
        #
        #         img_mask = Image.fromarray(mask_arr)
        #         plt.imsave('%d.png' % mask_name, img_mask) ###개수?세는거..
        #         plt.imshow(img_mask, 'gray')
        #         plt.show()
