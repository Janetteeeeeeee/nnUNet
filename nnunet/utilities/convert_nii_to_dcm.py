from rt_utils import RTStructBuilder
from SimpleITK import GetArrayFromImage, ReadImage
import matplotlib.pyplot as plt
import numpy as np

segmentation_file = r"E:\AutoSeg_Bladder_data\nnUnet\predict\fold1brachy\bladder_201.nii.gz"

# Create new RT Struct. Requires the DICOM series path for the RT Struct.
# rtstruct = RTStructBuilder.create_new(dicom_series_path="./testlocation")

# Load existing RT Struct. Requires the series path and existing RT Struct path
rtstruct = RTStructBuilder.create_from(
  dicom_series_path=r"E:\AutoSeg_Bladder_data\Dicom\Test\Oncentra_Bladder & Rectum\JIANG^YL^^^-001",
  rt_struct_path=r"E:\AutoSeg_Bladder_data\Dicom\Test\Oncentra_Bladder & Rectum\JIANG^YL^^^-001\RTSTRUCT_1.2.276.0.7230010.3.1.4.315511641.17156.1625388376.1001.dcm"
)

MASK_FROM_ML_MODEL = GetArrayFromImage(ReadImage(segmentation_file))
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
rtstruct.add_roi(
  mask=mask_boolean,
  color=[255, 0, 255],
  name="Bladder_predict"
)

rtstruct.save(r'C:\Users\Miner2\Desktop\new-rt-struct.dcm')
print()

