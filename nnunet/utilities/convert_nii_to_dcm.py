from rt_utils import RTStructBuilder
from SimpleITK import GetArrayFromImage, ReadImage
import matplotlib.pyplot as plt
import numpy as np

segmentation_file = r"I:\Data\convert-test\bladder_001.nii.gz"

# Create new RT Struct. Requires the DICOM series path for the RT Struct.
# rtstruct = RTStructBuilder.create_new(dicom_series_path="./testlocation")

# Load existing RT Struct. Requires the series path and existing RT Struct path
rtstruct = RTStructBuilder.create_from(
  dicom_series_path=r"I:\Data\convert-test\001",
  rt_struct_path=r"I:\Data\convert-test\001\RS1.3.6.1.4.1.2452.6.3152533160.1205408082.1646658237.2945066707.dcm"
)

MASK_FROM_ML_MODEL = GetArrayFromImage(ReadImage(segmentation_file))
mask_boolean = MASK_FROM_ML_MODEL > 0
mask_boolean = mask_boolean.swapaxes(0, 2)
mask_boolean = mask_boolean.swapaxes(0, 1)
mask_boolean = np.flip(mask_boolean, 0)
plt.imshow(mask_boolean[:, :, 35])
plt.show()
# ...
# Create mask through means such as ML
# ...

# Add another ROI, this time setting the color, description, and name
rtstruct.add_roi(
  mask=mask_boolean,
  color=[255, 0, 255],
  name="Bladder_predict"
)

rtstruct.save(r'I:\Data\convert-test\new-rt-struct.dcm')
print()

