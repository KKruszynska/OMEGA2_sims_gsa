import pandas as pd
import numpy as np

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import matplotlib.pyplot as plt
from matplotlib import gridspec

def exclude_KMTNet_fields(names, ra, dec):
    KMTNet_fields = pd.read_csv("kmtnet_zona.csv", header=0)
    remove_idx = []

    exclusion_zone = Polygon(zip(KMTNet_fields["ra"], KMTNet_fields["dec"]))
    for i in range(len(names)):
        point = Point(ra[i], dec[i])
        if(exclusion_zone.contains(point)):
            remove_idx.append(i)
            # print("%s: Within KMTNet exclusion zone."%(names[i]))

    return np.array(remove_idx)

def zona_test():
    KMTNet_fields = pd.read_csv("kmtnet_zona.csv", header=0)
    coord = np.hstack((KMTNet_fields["ra"].values.reshape(len(KMTNet_fields["ra"]), 1),
                         KMTNet_fields["dec"].values.reshape(len(KMTNet_fields["ra"]), 1)))
    coord = coord.tolist()
    coord.append(coord[0])


    file_name = "GSA_2016_2021_catalog.csv"
    gsa_events = pd.read_csv(file_name, header=0)
    xs, ys = zip(*coord)

    remove_idx = exclude_KMTNet_fields(gsa_events["#Name"], gsa_events["Ra_deg"], gsa_events["Dec_deg"])
    to_remove_ra, to_remove_dec = gsa_events["Ra_deg"].iloc[remove_idx], gsa_events["Dec_deg"].iloc[remove_idx]

    plt.figure()
    plt.scatter(gsa_events["Ra_deg"], gsa_events["Dec_deg"], s=10)
    plt.scatter(to_remove_ra, to_remove_dec, color="yellow", s=5)
    plt.plot(xs, ys, color='black', zorder=10)
    plt.show()

def get_blend_fluxes_mcmc_samples(pspl, samples):
    result = []
    walkers = len(samples[0])
    for walker in range(walkers):
        res_walk = []
        for idx in range(len(samples)):
            fsfbs_array = []
            pyLIMA_parameters = pspl.compute_pyLIMA_parameters(samples[idx][walker])

            for telescope in pspl.event.telescopes:
                model = pspl.compute_the_microlensing_model(telescope, pyLIMA_parameters)
                f_source = model['f_source']
                f_blend = model['f_blend']
                fsfbs_array.append(f_source)
                fsfbs_array.append(f_blend)
            res_walk.append(fsfbs_array)
        result.append(res_walk)

    result = np.array(result)
    return result

def save_walker_steps_plot(samples, blend_samples):
    # samples_without_ln_like = fit_gaia.fit_results["MCMC_chains"][:, :, :5]
    n_dim = len(samples[0,0,:])
    n_flux = len(blend_samples[0,0,:])
    n_steps = len(samples[:,0])
    n_walkers = len(samples[0])

    parameters = ["t_0", "u_0", "t_E", "pi_EN", "pi_EE"]

    fig = plt.figure(figsize=(10,14))
    # let's see how does the trace plot look like
    colors = plt.cm.jet(np.linspace(0, 1, n_dim + n_flux))
    steps = np.arange(n_steps)
    gs = gridspec.GridSpec(n_dim + n_flux, 1)

    for dim in range(n_dim):
        plt.subplot(gs[dim])
        for walk in range(n_walkers):
            plt.plot(steps, samples[:, walk, dim], color=colors[dim], lw=0.2)
            plt.ylabel(parameters[dim], fontsize=8)

    for dim in range(n_flux):
        plt.subplot(gs[n_dim + dim])
        for walk in range(n_walkers):
            plt.plot(steps, blend_samples[walk, :, dim], color=colors[dim], lw=0.2)
            if(dim % 2 == 0):
                plt.ylabel("F_s,%d" % (int(dim/2)), fontsize=8)
            else:
                plt.ylabel("F_b,%d" % (int(dim/2)), fontsize=8)

    plt.xlabel("Steps")

    return fig

# This one was defined by Markus...
def calculate_blend_and_source_flux(best_fit_process_parameters, your_model):
    result = []
    pyLIMA_parameters = your_model.compute_pyLIMA_parameters(best_fit_process_parameters)
    for telescope in your_model.event.telescopes:
        model = your_model.compute_the_microlensing_model(telescope, pyLIMA_parameters)
        f_source = model['f_source']
        f_blend = model['f_blend']
        result.append(f_source)
        result.append(f_blend)
    return result

def save_best_sol(file_name, name, your_model, best_params):
    # Figure out blend parameters
    blend_fluxes = calculate_blend_and_source_flux(best_params, your_model)

    param_names = ["t_0", "u_0", "t_E", "pi_EN", "pi_EE"]
    output = open(file_name, "w")
    output.write("%s : Parameters for best solution\n" % (name))
    for i in range(len(best_params)):
        output.write("%s = %.4f\n" % (param_names[i], best_params[i]))
    for i in range(len(blend_fluxes)):
        if( i % 2 == 0):
            output.write("Source flux for telescope %d = %.4f\n" % (int(i/2.), blend_fluxes[i]))
        else:
            output.write("Blend flux for telescope %d = %.4f\n" % (int(i / 2.), blend_fluxes[i]))
    output.close()