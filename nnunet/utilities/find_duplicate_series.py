import os

import pydicom
from batchgenerators.utilities.file_and_folder_operations import join, subfiles

if __name__ == '__main__':
    bladder_series_dir = r'E:\AutoSeg_Bladder_data\Dicom\Train&Valid'
    rectum_series_dir = r'E:\AutoSeg_Rectum_data\Train&Valid\6th_Eclips'
    series_list = os.listdir(rectum_series_dir)
    position_dict = {}
    for series in series_list:
        dicom_files = subfiles(join(rectum_series_dir, series), prefix='1.', suffix='.dcm')
        if len(dicom_files) > 0:
            dicom_file = pydicom.read_file(dicom_files[0])
            exposure_time = dicom_file[0x0018, 0x1150]
            position = dicom_file[0x0020, 0x0032].value
            if (position[0], position[1], position[2]) in position_dict:
                print(series)
                print(position_dict[(position[0], position[1], position[2])])
            else:
                position_dict[(position[0], position[1], position[2])] = series
