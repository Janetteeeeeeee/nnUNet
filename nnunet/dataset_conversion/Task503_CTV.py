import os
import shutil

import numpy as np
import cv2 as cv
from skimage import measure
import dicom2nifti
import nibabel as nib
import pydicom
from batchgenerators.utilities.file_and_folder_operations import join, isdir, subfiles, maybe_mkdir_p
from nnunet.paths import nnUNet_raw_data
from collections import OrderedDict
from nnunet.dataset_conversion.utils import generate_dataset_json


def read_ct_rs(ct_rs_folder):
    ct_rs_files = os.listdir(ct_rs_folder)
    ct_files = []
    rs_file = ""
    for i in range(ct_rs_files.__len__()):
        temp_file = ct_rs_files[i]
        ct_file_index = 0
        if temp_file.startswith('RS') or temp_file.startswith('RTSTRUCT'):
            rs_file = os.path.join(ct_rs_folder, temp_file)
        elif (temp_file.startswith('CT') or temp_file.endswith('.dcm')) and not (temp_file.startswith('RD') or temp_file.startswith('RP')):
            ct_files.append(os.path.join(ct_rs_folder, temp_file))
            ct_file_index += 1
    if rs_file == "":
        print("no rt structure file", ct_rs_folder)
    return ct_files, rs_file


def extract_contour(ct_files, rs_file):
    ct_dicom_info = pydicom.read_file(ct_files[0])
    rs_dicom_info = pydicom.read_file(rs_file)
    patient_image_position = ct_dicom_info.data_element('ImagePositionPatient').repval
    patient_image_position = str.strip(patient_image_position, '[]').split(',')
    pixel_spacing = ct_dicom_info.data_element('PixelSpacing').repval
    pixel_spacing = str.strip(pixel_spacing, '[]').split(',')
    img_length = int(ct_dicom_info.data_element('Columns').repval)
    img_width = int(ct_dicom_info.data_element('Rows').repval)
    img_height = ct_files.__len__()
    contour_splined = np.zeros((img_length, img_width, img_height), dtype=float)
    contour_filled = np.zeros((img_length, img_width, img_height), dtype=float)
    slice_location = []
    for i in range(ct_files.__len__()):
        try:
            temp_dicominfo = pydicom.read_file(ct_files[i])
        except:
            print("error file", ct_files[i])
        temp_slice_loc = temp_dicominfo.data_element('SliceLocation').value
        temp_slice_thickness = temp_dicominfo.data_element('SliceThickness').value
        if temp_slice_loc is not None:
            slice_location.append(int(temp_slice_loc))
        else:
            slice_location.append(slice_location[-1] + temp_slice_thickness)
    list.sort(slice_location)

    roi_list = []
    roi_name_index_dict = {}
    contour_search = ['ctv', 'gtv']
    for contour_s in contour_search:
        for index, roi in enumerate(rs_dicom_info[0x3006, 0x0020]):
            roi_name = roi[0x3006, 0x0026].value.lower()
            if roi_name.find(contour_s) >= 0:
                roi_name_index_dict[contour_s] = index
                break
    for contour_s in contour_search:
        if contour_s in roi_name_index_dict:
            roi_list.append(roi_name_index_dict[contour_s])
            break
        # elif roi_name.find('bladder-j') >= 0:
        #     roi_name_index_dict['bladder-j'] = index
        # elif roi_name.find('bladder') >= 0:
        #     roi_name_index_dict['bladder'] = index
    # elif 'bladder-j' in roi_name_index_dict:
    #     roi_list.append(roi_name_index_dict['bladder-j'])
    # elif 'bladder' in roi_name_index_dict:
    #     roi_list.append(roi_name_index_dict['bladder'])

    if len(roi_list) == 0:
        raise Exception

    roi_contour_sequence = rs_dicom_info.data_element('ROIContourSequence').value.__getattribute__('_list')
    for i in roi_list:
        try:
            temp_contour_sequence = roi_contour_sequence[i].data_element('ContourSequence')
        except:
            print('Contour sequence not found')

    contour_sequence = temp_contour_sequence.value._list
    for i in range(len(contour_sequence)):
        temp_contour_data = contour_sequence[i].data_element('ContourData').value._list
        temp_contour_x = np.array(temp_contour_data[0::3])
        temp_contour_y = np.array(temp_contour_data[1::3])
        temp_contour_z = np.array(temp_contour_data[2::3])
        temp_slice_loc = temp_contour_z[0]
        slice_idx = np.argmin(np.abs(np.array(slice_location) - temp_slice_loc))
        axis_x = np.abs(np.floor((temp_contour_x - float(patient_image_position[0])) / float(pixel_spacing[0])) + 1)
        axis_y = np.abs(np.floor((temp_contour_y - float(patient_image_position[1])) / float(pixel_spacing[1])) + 1)
        axis_x = np.append(axis_x, axis_x[0])
        axis_y = np.append(axis_y, axis_y[0])
        x_interval = np.abs(axis_x[1:] - axis_x[0:-1])
        y_interval = np.abs(axis_y[1:] - axis_y[0:-1])
        n_xinterval = (x_interval > float(pixel_spacing[0])) + (y_interval > float(pixel_spacing[1]))
        n_xinterval = np.where(n_xinterval)[0]
        for pos_i in n_xinterval:
            interval_num = int(np.floor(np.max([x_interval[pos_i], y_interval[pos_i]]) + 3))
            x_interval_point = np.linspace(start=axis_x[pos_i], stop=axis_x[pos_i + 1], num=interval_num)
            y_interval_point = np.linspace(start=axis_y[pos_i], stop=axis_y[pos_i + 1], num=interval_num)
            axis_x = np.append(axis_x, x_interval_point[1:-1])
            axis_y = np.append(axis_y, y_interval_point[1:-1])
        axis_x = np.floor(axis_x + 0.5)
        axis_y = np.floor(axis_y + 0.5)
        for jj in range(len(axis_x)):
            contour_splined[int(axis_x[jj]), int(axis_y[jj]), slice_idx] = 1

    for slice_i in range(ct_files.__len__()):
        if np.sum(contour_splined[:, :, slice_i]) != 0:
            dilate_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
            temp_contour = cv.dilate(np.uint8(contour_splined[:, :, slice_i]), dilate_kernel)
            if np.sum(temp_contour[:]) != 0:
                temp_contour = np.where(temp_contour > 0, 0, 1)
                temp_contour_labels = measure.label(temp_contour, background=0)
                temp_contour_filled = np.uint8(np.where(temp_contour_labels == 1, 0, 1))
                temp_contour_filled = cv.erode(temp_contour_filled, dilate_kernel)
                contour_filled[:, :, slice_i] = np.flipud(np.flip(temp_contour_filled))
    return contour_filled


def convert_file(mode, dicom_root_folder, task_id, task_name):
    task_root_folder = nnUNet_raw_data + '/Task' + task_id + '_' + task_name
    label_filename_suffix = '.nii.gz'
    maybe_mkdir_p(task_root_folder)
    if mode == 'train':
        train_image = join(task_root_folder, 'imagesTr')
        train_label = join(task_root_folder, 'labelsTr')
        if isdir(train_image):
            shutil.rmtree(train_image)
        if isdir(train_label):
            shutil.rmtree(train_label)
        maybe_mkdir_p(train_image)
        maybe_mkdir_p(train_label)
        patient_dir_list = os.listdir(dicom_root_folder)
        image_nii_folders = []
        for patient_dir in patient_dir_list:
            dicom_folder = os.path.join(dicom_root_folder, patient_dir)
            dicom_files, contour_file = read_ct_rs(dicom_folder)

            try:
                # save rt structure
                contour_filled = extract_contour(dicom_files, contour_file)
                contour_nii = nib.Nifti1Image(contour_filled, np.eye(4))
                nib.save(contour_nii, join(train_label, task_name + '_' + patient_dir + label_filename_suffix))

                # save image
                maybe_mkdir_p(join(train_image, patient_dir))
                dicom2nifti.convert_directory(dicom_folder, join(train_image, patient_dir), compression=True)
                image_nii_folders.append(patient_dir)
            except Exception as e:
                print(e)
                print('No contour ctv', patient_dir)

        for nii_folder in image_nii_folders:
            niftis = subfiles(join(train_image, nii_folder), suffix=".nii.gz")
            nii_gz_file = niftis[0]
            shutil.move(nii_gz_file, join(train_image, task_name + '_' + nii_folder + '_0000' + label_filename_suffix))
            # shutil.rmtree(join(train_image, nii_folder))

        # create json
        generate_dataset_json(join(task_root_folder, 'dataset.json'), train_image, None, ('CT',),
                              {0: 'background', 1: 'bladder'}, dataset_name=task_name)
    elif mode == 'test':
        test_image = join(task_root_folder, 'imagesTs')
        test_label = join(task_root_folder, 'labelsTs')
        # if isdir(test_image):
        #     shutil.rmtree(test_image)
        # if isdir(test_label):
        #     shutil.rmtree(test_label)
        maybe_mkdir_p(test_image)
        maybe_mkdir_p(test_label)
        patient_dir_list = os.listdir(dicom_root_folder)
        image_nii_folders = []
        for patient_dir in patient_dir_list:
            # if int(patient_dir) < 518:
            #     continue
            dicom_folder = os.path.join(dicom_root_folder, patient_dir)
            dicom_files, contour_file = read_ct_rs(dicom_folder)

            try:
                # save rt structure
                contour_filled = extract_contour(dicom_files, contour_file)
                contour_nii = nib.Nifti1Image(contour_filled, np.eye(4))
                nib.save(contour_nii, join(test_label, task_name + '_' + patient_dir + label_filename_suffix))

                # save image
                maybe_mkdir_p(join(test_image, patient_dir))
                dicom2nifti.convert_directory(dicom_folder, join(test_image, patient_dir), compression=True)
                image_nii_folders.append(patient_dir)
            except Exception as e:
                print(e)
                print('No contour ctv', patient_dir)

        for nii_folder in image_nii_folders:
            niftis = subfiles(join(test_image, nii_folder), suffix=".nii.gz")
            nii_gz_file = niftis[0]
            shutil.move(nii_gz_file, join(test_image, task_name + '_' + nii_folder + '_0000' + label_filename_suffix))
            shutil.rmtree(join(test_image, nii_folder))

if __name__ == "__main__":
    mode = 'test'
    task_id = '504'
    task_name = 'ctvfull'
    dicom_root_folder = r'E:\CTV\AutoSeg_CTV_data_164cases\train'
    convert_file(mode, dicom_root_folder, task_id, task_name)

