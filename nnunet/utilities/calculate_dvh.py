from dicompylercore import dicomparser, dvh, dvhcalc
from batchgenerators.utilities.file_and_folder_operations import join, isdir, subfiles, maybe_mkdir_p
import os
import csv


def printStructure(rs, rs_file, rd_file, patient, predict=False):
    structures = rs.GetStructures()
    rows = []
    headers = []
    firstrow = True
    for structure in structures.items():
        num = structure[0]
        name = structure[1]["name"]
        # print(name)
        # Calculate a DVH from DICOM RT data
        calcdvh = dvhcalc.get_dvh(rs_file, rd_file, num)
        # print("Volume\t" + str(calcdvh.volume) + " cm3")
        if predict:
            if name.lower().find('predict') < 0:
                continue

        if name.lower().find("ctv") >= 0:
            calcdvh.rx_dose = rx_dose
            # print("D90\t" + str(calcdvh.D90))
            # print("V100\t" + str(calcdvh.V100))
            # print("V150\t" + str(calcdvh.V150))
            # print("V200\t" + str(calcdvh.V200))
            header = ["patient", "RxDose", "OAR", "Volume(cm3)", "D90(Gy)", "V100(cm3)", "V150(cm3)", "V200(cm3)"]
            row = [patient, rx_dose, name, str(round(calcdvh.volume, 3)) + " cm3", str(calcdvh.D90), str(calcdvh.V100), str(calcdvh.V150), str(calcdvh.V200)]
            headers = headers + header
            rows = rows + row
        else:
            # print("D2CC\t" + str(calcdvh.D2CC))
            # print("D1CC\t" + str(calcdvh.D1CC))
            # print("D0.1CC\t" + str(calcdvh.cumulative.dose_constraint(0.1, "cc")))
            # print("DMax\t" + str(calcdvh.max) + " Gy")
            header = ["patient", "RxDose", "OAR", "Volume(cm3)", "D2CC(Gy)", "D1CC(Gy)", "D0.1CC(Gy)", "DMax(Gy)"]
            row = [patient, rx_dose, name, str(round(calcdvh.volume, 3)) + " cm3", str(calcdvh.D2CC), str(calcdvh.D1CC),
                   str(calcdvh.cumulative.dose_constraint(0.1, "cc")), str(calcdvh.max) + " Gy"]
            headers = headers + header
            rows = rows + row
    return rows


if __name__ == '__main__':
    dose_dir = r'/media/lu/Data/Data/Dicom/test/RD'
    predict_dcm_dir = r'/media/lu/Data/Data/Dicom/test/predict/final_data'
    csv_output_file = r'/media/lu/Data/Data/Dicom/test/RD_statistics.csv'
    rows = []
    for patient in os.listdir(dose_dir):
        print(patient)
        rd_file = subfiles(join(dose_dir, patient), prefix="RD")[0]
        rs_file = subfiles(join(dose_dir, patient), prefix="RS")[0]
        rs_predict_file = join(predict_dcm_dir, patient + "_predict.dcm")
        rs_predict = dicomparser.DicomParser(rs_predict_file)
        rs = dicomparser.DicomParser(rs_file)
        rd = dicomparser.DicomParser(rd_file)
        rp_uid = rd.GetReferencedRTPlan()
        rp = dicomparser.DicomParser(join(dose_dir, patient, "RP" + rp_uid + ".dcm"))
        rx_dose = 0
        for setup in rp.ds.FractionGroupSequence[0].ReferencedBrachyApplicationSetupSequence:
            rx_dose += setup.BrachyApplicationSetupDose
        # print("RxDose\t" + str(rx_dose) + " Gy")
        # i.e. Get a dict of structure information
        rows.append(printStructure(rs, rs_file, rd_file, patient, predict=False))
        rows.append(printStructure(rs_predict, rs_predict_file, rd_file, patient, predict=True))
        # calcdvh.relative_volume.describe()
        print()
    with open(csv_output_file, 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(rows)
