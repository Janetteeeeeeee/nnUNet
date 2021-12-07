from rt_utils import RTStructBuilder
from SimpleITK import GetArrayFromImage, ReadImage
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join


def convert_to_dcm(nii_dir: str, roi_name: str, dicom_dir: str, rtst_dir: str, update: bool):
    niis = subfiles(nii_dir, join=False, suffix="nii.gz")

    for nii in niis:
        nii_file = join(nii_dir, nii)
        patient_id = nii.split(".")[0].split("_")[1]
        dicom_series = join(dicom_dir, patient_id)
        if update:
            rt_struct_path = join(rtst_dir, patient_id + "_predict.dcm")
        else:
            rt_struct_path = join(dicom_series, subfiles(dicom_series, join=False, prefix="RS")[0])
        # Create new RT Struct. Requires the DICOM series path for the RT Struct.
        # rt_struct = RTStructBuilder.create_new(dicom_series_path=dicom_series)

        # Load existing RT Struct. Requires the series path and existing RT Struct path

        rt_struct = RTStructBuilder.create_from(dicom_series_path=dicom_series, rt_struct_path=rt_struct_path)

        predict_mask = GetArrayFromImage(ReadImage(nii_file))
        mask_boolean = predict_mask > 0
        mask_boolean = mask_boolean.swapaxes(0, 2)
        mask_boolean = mask_boolean.swapaxes(0, 1)
        mask_boolean = np.flip(mask_boolean, 0)
        # plt.imshow(mask_boolean[:, :, 35])
        # plt.show()
        # ...
        # Create mask through means such as ML
        # ...

        # Add another ROI, this time setting the color, description, and name
        try:
            if roi_name == "Bladder":
                rt_struct.add_roi(mask=mask_boolean, color=[255, 0, 255], name=roi_name+"_predict")
            if roi_name == "Rectum":
                rt_struct.add_roi(mask=mask_boolean, color=[0, 255, 255], name=roi_name + "_predict")
            if roi_name == "CTV":
                rt_struct.add_roi(mask=mask_boolean, color=[255, 255, 0], name=roi_name + "_predict")
        except Exception as e:
            print(patient_id)
            print(e)

        rt_struct.save(join(rtst_dir, patient_id + "_predict.dcm"))


if __name__ == '__main__':
    bladder_nii_dir = r"/media/lu/Data/Data/nnUnet/predict/bladder/190/ensemble"
    rectum_nii_dir = r"/media/lu/Data/Data/nnUnet/predict/rectum/3d_fullres/fold0"
    ctv_nii_dir = r"/media/lu/Data/Data/nnUnet/predict/ctvfull/ensemble"

    dicom_series_dir = r"/media/lu/Data/Data/Dicom/test/final_data"
    dicom_output_dir = r"/media/lu/Data/Data/Dicom/test/predict/final_data"

    convert_to_dcm(bladder_nii_dir, "Bladder", dicom_series_dir, dicom_output_dir, False)
    convert_to_dcm(rectum_nii_dir, "Rectum", dicom_series_dir, dicom_output_dir, True)
    convert_to_dcm(ctv_nii_dir, "CTV", dicom_series_dir, dicom_output_dir, True)
