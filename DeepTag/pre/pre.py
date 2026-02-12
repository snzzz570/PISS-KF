##（1）

import nibabel as nib
import os
dataroot = "/data2/zsn/Dataset/ACDC/database"

for root, dirs, files in os.walk(dataroot):
    for file in files:
        if file.endswith(".nii.gz"):
            img = nib.load(os.path.join(root, file))
            qform = img.get_qform()
            img.set_qform(qform)
            sfrom = img.get_sform()
            img.set_sform(sfrom)
            nib.save(img, os.path.join(root, file))

