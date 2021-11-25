from rt_utils import RTStructBuilder
from SimpleITK import GetArrayFromImage, ReadImage
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join

ensemble_prediction_dir = r"/media/lu/Data/Data/nnUnet/predict/bladder/190/ensemble"
test_dicom_dir = r"/media/lu/Data/Data/Dicom/test/final_data"
dicom_output_dir = r"/media/lu/Data/Data/Dicom/test/predict/final_data"

pred_nii = subfiles(ensemble_prediction_dir, join=False, suffix="nii.gz")

for pred in pred_nii:
    print(pred)
    nii_file = join(ensemble_prediction_dir, pred)
    patient_id = pred.split(".")[0].split("_")[1]
    dicom_series = join(test_dicom_dir, patient_id)
    rt_struct_path = subfiles(dicom_series, join=False, prefix="RS")[0]
    # Create new RT Struct. Requires the DICOM series path for the RT Struct.
    # rt_struct = RTStructBuilder.create_new(dicom_series_path=dicom_series)

    # Load existing RT Struct. Requires the series path and existing RT Struct path
    rt_struct = RTStructBuilder.create_from(dicom_series_path=dicom_series, rt_struct_path=join(dicom_series,
                                                                                                rt_struct_path))

    MASK_FROM_ML_MODEL = GetArrayFromImage(ReadImage(nii_file))
    mask_boolean = MASK_FROM_ML_MODEL > 0
    mask_boolean = mask_boolean.swapaxes(0, 2)
    mask_boolean = mask_boolean.swapaxes(0, 1)
    mask_boolean = np.flip(mask_boolean, 0)
    # plt.imshow(mask_boolean[:, :, 35])
    # plt.show()
    # ...
    # Create mask through means such as ML
    # ...

    # Add another ROI, this time setting the color, description, and name
    rt_struct.add_roi(
        mask=mask_boolean,
        color=[255, 0, 255],
        name="Bladder_predict"
    )

    rt_struct.save(join(dicom_output_dir, patient_id + "_predict.dcm"))

