import os

import numpy as np
import pandas as pd

import time
from datetime import datetime

from scipy.signal import savgol_filter

import multiprocessing as mul

import matplotlib.pyplot as plt
import bokeh.plotting  as bkp

from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA.fits import MCMC_fit
from pyLIMA.models import PSPL_model
from astropy.coordinates import SkyCoord
from astropy import units as u
from pyLIMA.outputs import pyLIMA_plots
from pyLIMA.outputs import file_outputs as pyLIMA_fo

import module_grab_data as mgd
import module_toolbox as mt

delta_t0 = 1.

# Open file with logs
log = open("OMEGA_MCMC_logs.log", "w")
log.write("%s : Start.\n" % (datetime.utcnow()))

# Load catalogue with all events + coordinates
file_name = "GSA_2016_2021_catalog.csv"
gsa_events = pd.read_csv(file_name, header = 0)

# Load all cross-match results for GSA and other surveys
fin = "xmatch_cat/xmatch_gsa_ogleews.csv"
ews_known = pd.read_csv(fin, header=0)
fin = "xmatch_cat/xmatch_gsa_ogle_mroz_bulge.csv"
ogle_known_A = pd.read_csv(fin, header=0)
fin = "xmatch_cat/xmatch_gsa_ogle_mroz_disc.csv"
ogle_known_B = pd.read_csv(fin, header=0)
fin = "xmatch_cat/xmatch_gsa_ogle_mroz_disc_possible.csv"
ogle_known_C = pd.read_csv(fin, header=0)
fin = "xmatch_cat/xmatch_gsa_moa.csv"
moa_known = pd.read_csv(fin, header=0)
fin = "xmatch_cat/xmatch_gsa_kmtnet.csv"
kmtnet_known = pd.read_csv(fin, header=0)
fin = "xmatch_cat/xmatch_gsa_asassn.csv"
asassn_known = pd.read_csv(fin, header=0)
fin = "xmatch_cat/xmatch_gsa_ztf_alerts.csv"
ztfal_known = pd.read_csv(fin, header=0)

# Exclude KMTNet events
kmtnet_exclusion_zone = mt.exclude_KMTNet_fields(gsa_events["#Name"], gsa_events["Ra_deg"], gsa_events["Dec_deg"])

#Time to run the loop...
for idx in range(0,20): #len(gsa_events["#Name"])):
    name = gsa_events["#Name"].values[idx]
    ra, dec = gsa_events["Ra_deg"].values[idx], gsa_events["Dec_deg"].values[idx]
    log.write("-----------------------------------------------------------------\n")
    log.write("%s : Procedure for %s started.\n" % (datetime.utcnow(), name))
    log.write("-----------------------------------------------------------------\n")

    if (idx in kmtnet_exclusion_zone):
        log.write("%s : %s : Target within KMTNet exclusion zone.\n" % (datetime.utcnow(), name))
        log.write("%s : Procedure for %s ended.\n" % (datetime.utcnow(), name))
        log.write("-----------------------------------------------------------------\n")
        continue

    if (name in ["Gaia16aye", "Gaia20fnr", "Gaia19dke"]):
        log.write("%s : %s : Target has way too much follow-up data.\n" % (datetime.utcnow(), name))
        log.write("%s : Procedure for %s ended.\n" % (datetime.utcnow(), name))
        log.write("-----------------------------------------------------------------\n")
        continue

    #Set up a pyLIMA event
    gaia_event = event.Event()
    gaia_event.name = name
    gaia_event.ra, gaia_event.dec = ra, dec
    log.write("%s : %s: Event crated.\n" % (datetime.utcnow(), name))

    event_names = name
    #Time to load data and add telescopes...
    # Gaia
    data = mgd.get_Gaia(name)
    telescope_gaia = telescopes.Telescope(name='Gaia',
                                          camera_filter='G',
                                          light_curve=data,
                                          light_curve_names=['time', 'mag', 'err_mag'],
                                          light_curve_units=['JD', 'mag', 'err_mag'],
                                          location='Space', spacecraft_name='Gaia')
    gaia_event.telescopes.append(telescope_gaia)
    log.write("%s : %s: Gaia data added.\n" % (datetime.utcnow(), name))

    # ASASSN
    res = asassn_known[asassn_known["name"] == name].index
    if(len(res) > 0):
        asassn_name = asassn_known["col1"].values[res[0]]
        event_names += ", %s" % (asassn_name)
        for band in ["g", "V"]:
            asassn_data = mgd.get_ASASSN(res, band, asassn_known)
            if (asassn_data is not int):
                telescope_asassn = telescopes.Telescope(name='ASASSN',
                                                       camera_filter=band,
                                                       light_curve=asassn_data,
                                                       light_curve_names=['time', 'mag', 'err_mag'],
                                                       light_curve_units=['JD', 'mag', 'err_mag'],
                                                        location='Earth')
                gaia_event.telescopes.append(telescope_asassn)
                log.write("%s : %s: ASASSN band %s data added.\n" % (datetime.utcnow(), name, band))

    # KMTNet
    res = kmtnet_known[kmtnet_known["name"] == name].index
    if(len(res) > 0):
        kmtn_name = kmtnet_known["name_2"].values[res[0]]
        event_names += ", %s" % (kmtn_name)
        for obs in ["KMTA", "KMTC", "KMTS"]:
            kmtn_data = mgd.get_KMTNet(res, name, obs, kmtnet_known)
            if (kmtn_data is not int):
                telescope_kmtn = telescopes.Telescope(name='KMTNet_%s' % (obs),
                                                      light_curve=kmtn_data,
                                                      light_curve_names=['time', 'flux', 'err_flux'],
                                                      light_curve_units=['JD', 'flux', 'err_flux'],
                                                      location='Earth')
                gaia_event.telescopes.append(telescope_kmtn)
                log.write("%s : %s: KMTNet data for site % added.\n" % (datetime.utcnow(), name, obs))

    # MOA
    res = moa_known[moa_known["name"] == name].index
    if (len(res) > 0):
        moa_name = moa_known["name_2"].values[res[0]]
        event_names += ", %s" % (moa_name)
        moa_data = mgd.get_MOA(res, name, moa_known)
        if (moa_data is not int):
            telescope_moa = telescopes.Telescope(name='MOA',
                                                 light_curve=moa_data,
                                                 light_curve_names=['time', 'flux', 'err_flux'],
                                                 light_curve_units=['JD', 'flux', 'err_flux'],
                                                 location='Earth')
            gaia_event.telescopes.append(telescope_moa)
            log.write("%s : %s: MOA data added.\n" % (datetime.utcnow(), name))

    #OGLE EWS
    res = ews_known[ews_known["name"] == name].index
    if (len(res) > 0):
        ews_name = ews_known["name_2"].values[res[0]]
        event_names += ", %s" % (ews_name)
        ews_data = mgd.get_OGLE_EWS(res, name, ews_known)
        if (ews_data is not int):
            telescope_ews = telescopes.Telescope(name='OGLE',
                                                 camera_filter='I',
                                                 light_curve=ews_data,
                                                 light_curve_names=['time', 'mag', 'err_mag'],
                                                 light_curve_units=['JD', 'mag', 'err_mag'],
                                                 location='Earth')
            gaia_event.telescopes.append(telescope_ews)
            log.write("%s : %s: OGLE EWS data added.\n" % (datetime.utcnow(), name))

    # OGLE Mróz et al. 2019a,b
    res = ogle_known_A[ogle_known_A["name"] == name].index
    if (len(res) > 0):
        ogle_name = ogle_known_A["col1"].values[res[0]]
        event_names += ", %s" % (ogle_name)
        ogle_data = mgd.get_OGLE_Mroz(res, ogle_known_A)
        if (ogle_data is not int):
            telescope_ogle = telescopes.Telescope(name='OGLE',
                                                  camera_filter='I',
                                                  light_curve=ogle_data,
                                                  light_curve_names=['time', 'mag', 'err_mag'],
                                                  light_curve_units=['JD', 'mag', 'err_mag'],
                                                  location='Earth')
            gaia_event.telescopes.append(telescope_ogle)
            log.write("%s : %s: OGLE Mroz data added.\n" % (datetime.utcnow(), name))

    res = ogle_known_B[ogle_known_B["name"] == name].index
    if (len(res) > 0):
        ogle_name = ogle_known_B["col1"].values[res[0]]
        event_names += ", %s" % (ogle_name)
        ogle_data = mgd.get_OGLE_Mroz(res, ogle_known_B)
        if (ogle_data is not int):
            telescope_ogle = telescopes.Telescope(name='OGLE',
                                                  camera_filter='I',
                                                  light_curve=ogle_data,
                                                  light_curve_names=['time', 'mag', 'err_mag'],
                                                  light_curve_units=['JD', 'mag', 'err_mag'],
                                                  location='Earth')
            gaia_event.telescopes.append(telescope_ogle)
            log.write("%s : %s: OGLE Mroz data added.\n" % (datetime.utcnow(), name))

    res = ogle_known_C[ogle_known_C["name"] == name].index
    if (len(res) > 0):
        ogle_name = ogle_known_C["col1"].values[res[0]]
        event_names += ", %s" % (ogle_name)
        ogle_data = mgd.get_OGLE_Mroz(res, ogle_known_C)
        if (ogle_data is not int):
            telescope_ogle = telescopes.Telescope(name='OGLE',
                                                  camera_filter='I',
                                                  light_curve=ogle_data,
                                                  light_curve_names=['time', 'mag', 'err_mag'],
                                                  light_curve_units=['JD', 'mag', 'err_mag'],
                                                  location='Earth')
            gaia_event.telescopes.append(telescope_ogle)
            log.write("%s : %s: OGLE Mroz data added.\n" % (datetime.utcnow(), name))

    # ZTF Alerts data
    res = ztfal_known[ztfal_known["#Name"] == name].index
    if (len(res) > 0):
        ztfal_name = ztfal_known["ZTFAlertName"].values[res[0]]
        event_names += ", %s" % (ztfal_name)
        for band in ["g", "r", "i"]:
            ztfal_data = mgd.get_ZTF_alerts(res, name, band, ztfal_known)
            if (type(ztfal_data) is not int):
                ztfal_data[:, 0] += 2450000.
                telescope_ztfal = telescopes.Telescope(name='ZTF',
                                                      camera_filter=band,
                                                      light_curve=ztfal_data,
                                                      light_curve_names=['time', 'mag', 'err_mag'],
                                                      light_curve_units=['JD', 'mag', 'err_mag'],
                                                       location='Earth')
                gaia_event.telescopes.append(telescope_ztfal)
                log.write("%s : %s: ZTF Alerts data in band %s added.\n" % (datetime.utcnow(), name, band))


    # ZTF DR data
    event_names += ", in ZTF DR"
    for band in ["g", "r", "i"]:
        ztf_data = mgd.get_ZTF(name, band)
        if (type(ztf_data) is not int):
            telescope_ztf = telescopes.Telescope(name='ZTF',
                                                 camera_filter=band,
                                                 light_curve=ztf_data,
                                                 light_curve_names=['time', 'mag', 'err_mag'],
                                                 light_curve_units=['JD', 'mag', 'err_mag'],
                                                 location='Earth')
            gaia_event.telescopes.append(telescope_ztf)
            log.write("%s : %s: ZTF DR data in band %s added.\n" % (datetime.utcnow(), name, band))


    # Follow-up data from BHTOM
    fup_dataframe = mgd.get_followup(name)
    if (type(fup_dataframe) is not int):
        event_names += ", has follow-up"
        for band in fup_dataframe["Filter"].unique():
            times = fup_dataframe[fup_dataframe["Filter"] == band]["JD"].values
            mags = fup_dataframe[fup_dataframe["Filter"] == band]["Magnitude"].values
            errs = fup_dataframe[fup_dataframe["Filter"] == band]["Error"].values
            fup_data = np.hstack((times.reshape(len(times),1),
                                  mags.reshape(len(times),1),
                                  errs.reshape(len(times),1)))

            fup_name = band.split('(')
            fup_name = "FUP_%s_%s" % (band[0],
                                      band.split("(")[1][:-1])
            telescope_fup = telescopes.Telescope(name=fup_name,
                                                 camera_filter=band[0],
                                                 light_curve=fup_data,
                                                 light_curve_names=['time', 'mag', 'err_mag'],
                                                 light_curve_units=['JD', 'mag', 'err_mag'],
                                                 location='Earth')
            gaia_event.telescopes.append(telescope_fup)
            log.write("%s : %s: Follow-up data in band %s added.\n" % (datetime.utcnow(), name, band))

    # Finally, lets check if the event looks fine...
    gaia_event.find_survey('Gaia')
    gaia_event.check_event()

    # initial guess for fitting
    data_smooth = savgol_filter(data[:, 1], 14, 4)
    t0guess = data[:, 0][np.argmin(data_smooth)]
    u0guess = 10. ** (-0.4 * (np.median(data_smooth) - np.min(data_smooth)))
    guess = [t0guess, u0guess, 30]
    pspl = PSPL_model.PSPLmodel(gaia_event, parallax=['Full', t0guess])

    # Time to start saving the outputs
    output_path = "outputs/%s" % (name)
    pathExist = os.path.exists(output_path)
    if not pathExist:
        os.makedirs(output_path)
    # test before start
    plot_start = pyLIMA_plots.plot_lightcurves(pspl, guess)
    # save figure
    plot_start[0].savefig('%s/%s_start.png' % (output_path, name),
                          format="png")
    # # save bokeh plot
    # bkp.output_file(filename="%s/%s_start.html" % (path, name),
    #             title="%s starting plot" % (name))
    # bkp.save(plot_start[1])

    # Time to start fitting...
    log.write("%s : %s : Setting up MCMC.\n" % (datetime.utcnow(), name))
    fit_gaia = MCMC_fit.MCMCfit(pspl)

    # starting point, can be adjusted
    fit_gaia.model_parameters_guess = [t0guess, u0guess, 60, 0.1, 0.1]

    # boundries for MCMC, can be adjusted
    fit_gaia.fit_parameters["t0"][1] = [t0guess - delta_t0, t0guess + delta_t0]
    fit_gaia.fit_parameters["u0"][1] = [-2., 2.]
    fit_gaia.fit_parameters["tE"][1] = [3., 3000.]
    fit_gaia.fit_parameters["piEE"][1] = [-2, 2.]
    fit_gaia.fit_parameters["piEN"][1] = [-2, 2.]

    # starting fit
    pool = mul.Pool(processes=7)
    log.write("%s : %s : MCMC start.\n" % (datetime.utcnow(), name))
    t_start = time.time()
    fit_gaia.fit(computational_pool = pool)
    t_end = time.time()
    log.write("%s : %s : MCMC end, fitting took %.2f seconds.\n" % (datetime.utcnow(), name, t_end - t_start))

    # Time to save outputs
    # Save samples with ln like
    sample_output = "%s/%s_mcmc_samples.npy" % (output_path, name)
    np.save(sample_output, fit_gaia.fit_results["MCMC_chains"])

    # Obtaining blend and source fluxes...
    samples_without_ln_like = fit_gaia.fit_results["MCMC_chains"][:, :, :5]
    blend_result = mt.get_blend_fluxes_mcmc_samples(pspl, samples_without_ln_like)

    # Saving blend fluxes
    blend_output = "%s/%s_mcmc_sample_blends.npy" % (output_path, name)
    np.save(blend_output, blend_result)

    # Time to save diagnostic plots
    # Creating and saving walkers path plot
    figure_steps = mt.save_walker_steps_plot(samples_without_ln_like, blend_result)
    fig_output = "%s/%s_mcmc_walker_path_plot.pdf" % (output_path, name)
    figure_steps.savefig(fig_output, format="pdf")

    # Best model plot
    pyLIMA_plots.list_of_fake_telescopes = []  # for replotting
    plot_end = pyLIMA_plots.plot_lightcurves(pspl, fit_gaia.fit_results['best_model'])
    plot_end[0].savefig('%s/%s_end.pdf' % (output_path, name), format="pdf")

    plot_geo_end = pyLIMA_plots.plot_geometry(pspl, fit_gaia.fit_results['best_model'])
    plot_geo_end[0].savefig('%s/%s_geo_end.pdf' % (output_path, name), format="pdf")

    # pyLIMA_fo.pdf_output(fit_gaia, output_path)

    # Save best solution
    best_output = "%s/%s_best_solution.txt" % (output_path, name)
    mt.save_best_sol(best_output, name, pspl, fit_gaia.fit_results['best_model'])

    # Finally, we're done with this event. Time to commemorate it in the log...
    log.write("%s : Procedure for %s ended.\n" % (datetime.utcnow(), name))
    log.write("-----------------------------------------------------------------\n")
log.close()