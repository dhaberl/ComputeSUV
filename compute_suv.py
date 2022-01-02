#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SUV CALCULATOR
Contains computation of Standardized Uptake Values based on body weight (SUVbw)
along with its utility functions.
Date: 2022-01-02 18:12:53
Author: dhaberl
Reference:
https://github.com/mvallieres/radiomics/blob/master/Utilities/computeSUVmap.m
"""

from datetime import datetime
import numpy as np
import pydicom
import pandas as pd


def _assert_time_format(time):

    time = time.split('.')[0]
    time_format = '%H%M%S'
    time = datetime.strptime(time, time_format)

    return time


def compute_suvbw(img, weight, scan_time, injection_time, half_life, injected_dose):
    """Compute SUV map based on given weight and injected dose decay."""

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
        assert np.isnan(i) == False, f'Invalid input. No NaNs allowed. Value is NaN: {np.isnan(i)}'

    assert weight < 1000, 'Weight exceeds 1000 kg, did you really used kg unit?'

    img = np.asarray(img)

    # Calculate decay
    decay = np.exp(-np.log(2) * time_difference / half_life)
    # Calculate the dose decayed during procedure in [Bq]
    injected_dose_decay = injected_dose * decay

    # Weight in grams
    weight = weight * 1000

    # SUVbw in g/ml
    suv_map = img * weight / injected_dose_decay

    return suv_map


def get_dicom_tags(dcm):
    """Return required information for SUV calculation"""

    # Ensure input parameter validity
    assert dcm.Modality == 'PT', 'Passed DICOM file is not a Positron-Emission-Tomography scan. Check DICOM Modality Tag.'

    # Get Patient Age
    try:
        age = dcm.PatientAge
    except AttributeError:
        print('Age is not stored in DICOM file.')
        age = np.nan

    # Get Patient Sex
    try:
        sex = dcm.PatientSex
    except AttributeError:
        print('Sex is not stored in DICOM file.')
        sex = np.nan

    # Get Patient Weight
    try:
        weight = dcm.PatientWeight
    except AttributeError:
        print('Weight is not stored in DICOM file.')
        weight = np.nan

    # Get Radiopharmaceutical Information
    try:
        tracer = dcm.RadiopharmaceuticalInformationSequence[0].Radiopharmaceutical
    except AttributeError:
        print('Radiopharmaceutical Info is not stored in DICOM file.')
        tracer = np.nan

    # Get Scan Time
    try:
        scan_time = dcm.AcquisitionTime
    except AttributeError:
        print('Acquisition Time is not stored in DICOM file.')
        scan_time = np.nan

    # Get Start Time for the Radiopharmaceutical Injection
    try:
        injection_time = dcm.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    except AttributeError:
        print('Injection Time is not stored in DICOM file.')
        injection_time = np.nan

    # Get Half Life for Radionuclide
    try:
        half_life = dcm.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    except AttributeError:
        print('Half Life is not stored in DICOM file.')
        half_life = np.nan

    # Get Total dose injected for Radionuclide
    try:
        injected_dose = dcm.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    except AttributeError:
        print('Injected Dose is not stored in DICOM file.')
        injected_dose = np.nan

    return {'age': age, 'sex': sex, 'weight': weight, 'tracer': tracer, 'scan_time': scan_time,
            'injection_time': injection_time, 'half_life': half_life, 'injected_dose': injected_dose}


def print_dicom_report(dcms, uids, save_as=None):

    df = {'id': [], 'age': [], 'sex': [], 'weight': [], 'tracer': [], 'scan_time': [],
          'injection_time': [], 'half_life': [], 'injected_dose': []}

    for i in zip(dcms, uids):
        tags = get_dicom_tags(i[0])
        df['id'].append(i[1])

        for tag in tags.keys():
            df[tag].append(tags[tag])

    df = pd.DataFrame(df)
    print(df)

    if save_as:
        df.to_csv(save_as, index=False)

    return df
