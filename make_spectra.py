import utils
import numpy as np
from picaso import justdoit as jdi
from scipy.stats import distributions
from scipy import optimize
from scipy.stats import norm
import os
import planets
import pickle
import yaml

def compute_spectra(add_water_cloud, add_haze, outfile):
    filename_db = os.path.join(os.getenv('picaso_refdata'), 'opacities','opacities.db')
    opa = jdi.opannection(wave_range=[.01,100],filename_db=filename_db)
    case1 = jdi.inputs()
    case1.phase_angle(0)
    case1.gravity(mass=planets.k2_18b.mass, mass_unit=jdi.u.Unit('M_earth'),
                radius=planets.k2_18b.radius, radius_unit=jdi.u.Unit('R_earth'))
    case1.star(opa, planets.k2_18.Teff, planets.k2_18.metal, planets.k2_18.logg, radius=planets.k2_18.radius, 
            radius_unit = jdi.u.Unit('R_sun'),database='phoenix')
    case1.approx(p_reference=1.0)

    model_type = ['habitable','habitable','habitable','neptune']
    model_names = ['noCH4','withCH4','withCH4_vdepCO','nominal']
    model_folders = [
        'results/habitable/',
        'results/habitable/',
        'results/habitable/',
        'results/neptune/'
    ]

    species_to_exclude = [['H2O'],['NH3'],['CO2'],['CH4'],['CO'],['HCN'],['H2O','NH3'],['H2O','NH3','CO']]
    res = {}
    for i in range(len(model_folders)):
        atmosphere_file = model_folders[i]+model_names[i]+'_picaso.pt'
        case1.atmosphere(filename = atmosphere_file, delim_whitespace=True)

        if add_water_cloud:
            # Get cloud region from settings file
            if model_type[i] == 'habitable':
                settings_file = model_folders[i]+model_names[i]+'_settings.yaml'
            else:
                settings_file = model_folders[i]+model_names[i]+'_settings_photochem.yaml'
            with open(settings_file,'r') as f:
                settings = yaml.load(f,Loader=yaml.Loader)

            # Add the cloud
            p_cloud_base = np.log10(settings['clouds']['P-condense']/1e6)
            p_coud_top = np.log10(settings['clouds']['P-trop']/1e6)
            cloud_thickness = p_cloud_base - p_coud_top
            case1.clouds(g0=[0.9], w0=[0.9], opd=[10], p=[p_cloud_base], dp=[cloud_thickness])

        if add_haze:
            case1.clouds(filename=model_folders[i]+model_names[i]+'_haze.txt', delim_whitespace=True)

        df = case1.spectrum(opa, full_output=True,calculation='transmission')
        wno_h, rprs2_h  = df['wavenumber'] , df['transit_depth']
        entry = {}
        entry['all'] = {}
        entry['all']['wv'] = 1e4/wno_h[::-1].copy()
        entry['all']['rprs2'] = rprs2_h[::-1].copy()
        for sp in species_to_exclude:
            case1.atmosphere(filename = atmosphere_file,exclude_mol=sp, delim_whitespace=True)
            df = case1.spectrum(opa, full_output=True,calculation='transmission')
            wno_h, rprs2_h  = df['wavenumber'] , df['transit_depth']
            key = '_'.join(sp)
            entry[key] = {}
            entry[key]['wv'] = 1e4/wno_h[::-1].copy()
            entry[key]['rprs2'] = rprs2_h[::-1].copy()

        res[model_names[i]] = entry

    with open(outfile,'wb') as f:
        pickle.dump(res,f)

def stats_objective(x, data_y, err, expected_y):
    return utils.chi_squared(data_y, err, expected_y+x[0])

def compute_statistics(infile, out_stats_file):

    i_values = [0,6]

    with open('data/osfstorage-archive/lowres.pkl','rb') as f:
        data = pickle.load(f)

    with open(infile,'rb') as f:
        models = pickle.load(f)

    models_binned = {}
    for i in i_values:
        # rebin models to data
        models_r = {}
        for model in models:
            models_r[model] = {}
            for case in models[model]:
                
                # regrid spectrum to data
                _, _, rprs2 = utils.rebin_picaso_to_data(models[model][case]['wv'], models[model][case]['rprs2'], data['all']['wv_bins'])

                # Save results
                models_r[model][case] = {}
                models_r[model][case]['wv'] = data['all']['wv']
                models_r[model][case]['rprs2'] = rprs2
                
                # Find offset
                init = np.array([1e-5])
                args = (data['all']['rprs2'][i:], data['all']['rprs2_err'][i:], rprs2[i:])
                sol = optimize.minimize(stats_objective, init, method = 'Nelder-Mead', args = args)
                assert sol.success
                models_r[model][case]['offset'] = sol.x[0]

                dof = data['all']['wv'][i:].shape[0]
                chi2 = utils.chi_squared(data['all']['rprs2'][i:], data['all']['rprs2_err'][i:], rprs2[i:]+sol.x[0])
                rchi2 = chi2/dof
                p = distributions.chi2.sf(chi2, dof)
                sig = norm.ppf(1 - p)
                models_r[model][case]['rchi2'] = rchi2
                models_r[model][case]['p'] = p
                models_r[model][case]['sig'] = sig

        model = 'flat'
        case = 'all'
        models_r[model] = {}
        rprs2[:] = 0.002944
        models_r[model][case] = {}
        models_r[model][case]['wv'] = data['all']['wv']
        models_r[model][case]['rprs2'] = rprs2

        init = np.array([1e-5])
        args = (data['all']['rprs2'][i:], data['all']['rprs2_err'][i:], rprs2[i:])
        sol = optimize.minimize(stats_objective, init, method = 'Nelder-Mead', args = args)
        assert sol.success
        models_r[model][case]['offset'] = sol.x[0]

        dof = data['all']['wv'][i:].shape[0]
        chi2 = utils.chi_squared(data['all']['rprs2'][i:], data['all']['rprs2_err'][i:], rprs2[i:]+sol.x[0])
        rchi2 = chi2/dof
        p = distributions.chi2.sf(chi2, dof)
        sig = norm.ppf(1 - p)
        models_r[model][case]['rchi2'] = rchi2
        models_r[model][case]['p'] = p
        models_r[model][case]['sig'] = sig

        models_binned[i] = models_r

    with open(out_stats_file,'wb') as f:
        pickle.dump(models_binned,f)

if __name__ == '__main__':
    add_water_cloud = False
    add_haze = False
    outfile = 'results/spectra/spectra.pkl'
    out_stats_file = 'results/spectra/spectra_stats.pkl'
    compute_spectra(add_water_cloud, add_haze, outfile)
    compute_statistics(outfile, out_stats_file)

    add_water_cloud = True
    add_haze = False
    outfile = 'results/spectra/spectra_watercloud.pkl'
    out_stats_file = 'results/spectra/spectra_watercloud_stats.pkl'
    compute_spectra(add_water_cloud, add_haze, outfile)
    compute_statistics(outfile, out_stats_file)

    add_water_cloud = False
    add_haze = True
    outfile = 'results/spectra/spectra_haze.pkl'
    out_stats_file = 'results/spectra/spectra_haze_stats.pkl'
    compute_spectra(add_water_cloud, add_haze, outfile)
    compute_statistics(outfile, out_stats_file)
    







