import numpy as np
import pandas as pd

import os

def get_ASASSN(res, band, asassn_known):
    '''
    Grab data from ASASSN saved in the repo.
    :param name: name of the alert in GSA
    :param band: band in which alert was observed
    :param asassn_known: dataframe with all known GSA events found in ASASSN
    :return: returns either an array with times, mags and errs
            or 0 (if file was not found).
    '''
    phot_name = asassn_known["phot_file"].values[res[0]]
    file_name = "data/asassn_data/%s_%s.dat" % (phot_name, band)
    if os.path.isfile(file_name):
        data = np.loadtxt(file_name, dtype=float, usecols=(0, 1, 2), unpack=True)
        return data
    else:
        return 0


def get_Gaia(name):
    '''
    Grab GSA data for the event.
    :param name:
    :return:
    '''
    file_name = "data/GSA/%s_%s.dat" % (name, name)
    data = np.loadtxt(file_name, dtype=float, usecols=(0, 1, 2))
    return data


def get_KMTNet(res, name, obs, kmtnet_known):
    '''
    Grab KMTNet data, for a given site.
    :param name:
    :param obs:
    :param kmtnet_known:
    :return:
    '''
    kmtn_name = kmtnet_known["name_2"].values[res[0]]
    file_name = "data/KMTNet/%s_%s_%s.dat" % (name, kmtn_name, obs)
    if os.path.isfile(file_name):
        data = np.loadtxt(file_name, dtype=float, usecols=(0, 1, 2))
        return data
    else:
        return 0


def get_MOA(res, name, moa_known):
    '''
    Grab MOA data.
    :param name:
    :param moa_known:
    :return:
    '''
    moa_name = moa_known["name_2"].values[res[0]]
    data = np.loadtxt("data/MOA/%s_%s.dat" % (name, moa_name), dtype=float, usecols=(0, 1, 2))
    return data


def get_OGLE_EWS(res, name, ogle_ews):
    '''
    Grab OGLE EWS data.
    :param name:
    :param ogle_ews:
    :return:
    '''
    ogle_name = ogle_ews["name_2"].values[res[0]]
    data = np.loadtxt("data/OGLE_EWS/%s_%s.dat" % (name, ogle_name),
                      dtype=float, usecols=(0, 1, 2))
    return data


def get_OGLE_Mroz(res, ogle_known):
    '''
    Grab OGLE data from Mróz et al. 2019 a, b
    (Gal bulge and disc optical dentisty maps).
    :param name:
    :param ogle_known:
    :return:
    '''
    ogle_name = ogle_known["col1"].values[res[0]]
    file_name = "data/ogle_data/" + ogle_name + ".dat"
    data = np.loadtxt(file_name, dtype=float, usecols=(0, 1, 2))
    return data


def get_ZTF_alerts(res, name, band, ztf_known):
    '''
    Grab ZTF alerts data.
    :param name:
    :param band:
    :param ztf_known:
    :return:
    '''
    ztfal_name = ztf_known["ZTFAlertName"].values[res[0]]
    file_name = "data/ZTFAlerts/%s_ZTFAl_%s_ZTF_al_%s.dat" % (name, ztfal_name, band)
    if os.path.isfile(file_name):
        data = np.loadtxt(file_name)
        return data
    else:
        return 0


def get_ZTF(name, band):
    '''
    Grab ZTF Data Release data (from ~08.2022).
    :param name:
    :param band:
    :return:
    '''
    file_name = "data/ZTF/%s_ZTF_%s.dat" % (name, band)
    if os.path.isfile(file_name):
        data = np.loadtxt(file_name)
        return data
    else:
        return 0

def get_followup(name):
    '''
    Grab follow-up data stored on BHTOM/CPCS2.
    Achtung! Returns a pandas dataframe!
    :param name:
    :return:
    '''
    file_name = "data/BHTOM/%s.csv" % (name)
    if os.path.isfile(file_name):
        data = pd.read_csv(file_name, header=0, delimiter=';')
        return data
    else:
        return 0