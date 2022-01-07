"""
COMPUTATION OF STANDARDIZED UPTAKE VALUES BASED ON BODY WEIGHT (SUVbw)
Reference:
[1] https://github.com/mvallieres/radiomics/blob/master/Utilities/computeSUVmap.m
"""

from datetime import datetime
import numpy as np
import pydicom
import pandas as pd


def _assert_time_format(time):
    """
    Time stamp formatting

    Args:
        time (str): Time stamp from DICOM file.

    Returns:
        time: datetime object
    """
    # Cut off milliseconds
    time = time.split('.')[0]
    time_format = '%H%M%S'
    time = datetime.strptime(time, time_format)

    return time


def compute_suvbw_map(img, weight, scan_time, injection_time, half_life, injected_dose):
    """
    Compute SUVbw map based on given weight and injected dose decay.

    Args:
        img: Input image ndarray. Each pixel/voxel is associated with its radioactivity
        represented as volume concentration MBq/mL. 
        weight: Patient body weight in kilograms.
        scan_time (str): Acquisition time (start time of PET). Time stamp from DICOM file.
        injection_time (str): Injection time; time when radiopharmaceutical dose was administered.
        Time stamp from DICOM file.
        half_life: Half life of used radiopharmaceutical in seconds.
        injected_dose: Injected total dose of administered radiopharmaceutical in Mega Becquerel.

    Returns:
        suv_map: Image ndarray. Each pixel/voxel is associated with its SUVbw.
    """

    # Assert time format
    scan_time = _assert_time_format(scan_time)
    injection_time = _assert_time_format(injection_time)
    # Calculate time in seconds between acqusition time (scan time) and injection time
    time_difference = scan_time - injection_time
    time_difference = time_difference.seconds

    # Ensure parameter validity
    check = [weight, time_difference, half_life, injected_dose]
    for i in check:
        assert i > 0, f'Invalid input. No negative values allowed. Value: {i}'
        assert np.isnan(
            i) == False, f'Invalid input. No NaNs allowed. Value is NaN: {np.isnan(i)}'

    assert weight < 1000, 'Weight exceeds 1000 kg, did you really used kg unit?'

    img = np.asarray(img)

    # Calculate decay for decay correction
    decay = np.exp(-np.log(2) * time_difference / half_life)
    # Calculate the dose decayed during procedure in Bq
    injected_dose_decay = injected_dose * decay

    # Weight in grams
    weight = weight * 1000

    # Calculate SUVbw
    suv_map = img * weight / injected_dose_decay

    return suv_map


def get_dicom_tags(dcm):
    """
    Return informative and required DICOM tags for SUV calculation. Missing DICOM tags will be returned as NaNs.
    Note: sex and age is not required but can help for estimations if values are missing (e.g. body weight)

    DICOM tags:
    https://dicom.innolitics.com/ciods

    Args:
        dcm (pydicom.dataset.FileDataset): Loaded DICOM file.
        Example:
            dcm = pydicom.dcmread(path_to_dcm_file)

        pydicom:
        https://pydicom.github.io/pydicom/stable/old/ref_guide.html

    Returns:
        dict: Dictionary with DICOM tags.
    """

    # Ensure input parameter validity
    assert dcm.Modality == 'PT', 'Passed DICOM file is not a Positron-Emission-Tomography scan. Check DICOM Modality tag.'

    # Get patient age
    try:
        age = dcm.PatientAge
    except AttributeError:
        print('Age is not stored in DICOM file.')
        age = np.nan

    # Get patient sex
    try:
        sex = dcm.PatientSex
    except AttributeError:
        print('Sex is not stored in DICOM file.')
        sex = np.nan

    # Get patient weight
    try:
        weight = dcm.PatientWeight
    except AttributeError:
        print('Weight is not stored in DICOM file.')
        weight = np.nan

    # Get radiopharmaceutical information (radiotracer)
    try:
        tracer = dcm.RadiopharmaceuticalInformationSequence[0].Radiopharmaceutical
    except AttributeError:
        print('Radiopharmaceutical Info is not stored in DICOM file.')
        tracer = np.nan

    # Get scan time
    try:
        scan_time = dcm.AcquisitionTime
    except AttributeError:
        print('Acquisition Time is not stored in DICOM file.')
        scan_time = np.nan

    # Get start time of the radiopharmaceutical injection
    try:
        injection_time = dcm.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    except AttributeError:
        print('Injection Time is not stored in DICOM file.')
        injection_time = np.nan

    # Get half life of radionuclide
    try:
        half_life = dcm.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    except AttributeError:
        print('Half Life is not stored in DICOM file.')
        half_life = np.nan

    # Get total dose injected for radionuclide
    try:
        injected_dose = dcm.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    except AttributeError:
        print('Injected Dose is not stored in DICOM file.')
        injected_dose = np.nan

    return {'age': age, 'sex': sex, 'weight': weight, 'tracer': tracer, 'scan_time': scan_time,
            'injection_time': injection_time, 'half_life': half_life, 'injected_dose': injected_dose}


def print_dicom_report(dcms, uids, save_as=None):
    """
    Prints DICOM tag report of a list of DICOM files

    Args:
        dcms (list): List of DICOM files loaded as pydicom.dataset.FileDataset.
        uids (list): List of unique IDs for each DICOM file.
        save_as (str, optional): Filename. Defaults to None.

    Returns:
        dicom_report (dict): DICOM report as dictionary.
    """
    dicom_report = {'id': [], 'age': [], 'sex': [], 'weight': [], 'tracer': [], 'scan_time': [],
                    'injection_time': [], 'half_life': [], 'injected_dose': []}

    for i in zip(dcms, uids):
        tags = get_dicom_tags(i[0])
        dicom_report['id'].append(i[1])

        for tag in tags.keys():
            dicom_report[tag].append(tags[tag])

    df = pd.DataFrame(dicom_report)
    print(df)

    if save_as:
        df.to_csv(save_as, index=False)

    return dicom_report
