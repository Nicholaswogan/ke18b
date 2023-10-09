import utils
import numpy as np
from picaso import justdoit as jdi
from scipy.stats import distributions
from scipy import optimize
from scipy.stats import norm
import os
import planets
import pickle

def compute_spectra():
    filename_db = os.path.join(os.getenv('picaso_refdata'), 'opacities','opacities.db')
    opa = jdi.opannection(wave_range=[.01,100],filename_db=filename_db)
    case1 = jdi.inputs()
    case1.phase_angle(0)
    case1.gravity(mass=planets.k2_18b.mass, mass_unit=jdi.u.Unit('M_earth'),
                radius=planets.k2_18b.radius, radius_unit=jdi.u.Unit('R_earth'))
    case1.star(opa, planets.k2_18.Teff, planets.k2_18.metal, planets.k2_18.logg, radius=planets.k2_18.radius, 
            radius_unit = jdi.u.Unit('R_sun'),database='phoenix')
    case1.approx(p_reference=1.0)

    model_names = ['noCH4','withCH4','withCH4_vdepCO','nominal','case1']
    atmosphere_files = [
        'results/habitable/noCH4_picaso.pt',
        'results/habitable/withCH4_picaso.pt',
        'results/habitable/withCH4_vdepCO_picaso.pt',
        'results/neptune/nominal_picaso.pt',
        'results/neptune/case1_picaso.pt'
    ]
    species_to_exclude = [['H2O'],['NH3'],['CO2'],['CH4'],['CO'],['HCN'],['H2O','NH3'],['H2O','NH3','CO']]
    res = {}
    for i,atmosphere_file in enumerate(atmosphere_files):
        case1.atmosphere(filename = atmosphere_file, delim_whitespace=True)
        df = case1.spectrum(opa, full_output=True,calculation='transmission')
        wno_h, rprs2_h  = df['wavenumber'] , df['transit_depth']
        entry = {}
        entry['all'] = {}
        entry['all']['wno'] = wno_h
        entry['all']['rprs2'] = rprs2_h
        for sp in species_to_exclude:
            case1.atmosphere(filename = atmosphere_file,exclude_mol=sp, delim_whitespace=True)
            df = case1.spectrum(opa, full_output=True,calculation='transmission')
            wno_h, rprs2_h  = df['wavenumber'] , df['transit_depth']
            key = '_'.join(sp)
            entry[key] = {}
            entry[key]['wno'] = wno_h
            entry[key]['rprs2'] = rprs2_h

        res[model_names[i]] = entry

    with open('results/spectra/spectra.pkl','wb') as f:
        pickle.dump(res,f)

def stats_objective(x, data_y, err, expected_y):
    return utils.chi_squared(data_y, err, expected_y+x[0])

def compute_statistics():

    i_values = [0,6]

    with open('data/data_fig.pkl','rb') as f:
        data = pickle.load(f)

    with open('results/spectra/spectra.pkl','rb') as f:
        models = pickle.load(f)

    models_binned = {}
    for i in i_values:
        # rebin models to data
        models_r = {}
        for model in models:
            models_r[model] = {}
            for case in models[model]:
                wno, rprs2 = jdi.mean_regrid(models[model][case]['wno'], models[model][case]['rprs2'], newx=1e4/data['all']['wv'][::-1])
                models_r[model][case] = {}
                models_r[model][case]['wno'] = wno
                models_r[model][case]['rprs2'] = rprs2

                init = np.array([1e-5])
                args = (data['all']['rprs2'][i:], data['all']['err'][i:], rprs2[::-1][i:])
                sol = optimize.minimize(stats_objective, init, method = 'Nelder-Mead', args = args)
                assert sol.success
                models_r[model][case]['offset'] = sol.x[0]

                dof = data['all']['wv'][i:].shape[0]
                chi2 = utils.chi_squared(data['all']['rprs2'][i:], data['all']['err'][i:], rprs2[::-1][i:]+sol.x[0])
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
        models_r[model][case]['wno'] = wno
        models_r[model][case]['rprs2'] = rprs2

        init = np.array([1e-5])
        args = (data['all']['rprs2'][i:], data['all']['err'][i:], rprs2[::-1][i:])
        sol = optimize.minimize(stats_objective, init, method = 'Nelder-Mead', args = args)
        assert sol.success
        models_r[model][case]['offset'] = sol.x[0]

        dof = data['all']['wv'][i:].shape[0]
        chi2 = utils.chi_squared(data['all']['rprs2'][i:], data['all']['err'][i:], rprs2[::-1][i:]+sol.x[0])
        rchi2 = chi2/dof
        p = distributions.chi2.sf(chi2, dof)
        sig = norm.ppf(1 - p)
        models_r[model][case]['rchi2'] = rchi2
        models_r[model][case]['p'] = p
        models_r[model][case]['sig'] = sig

        models_binned[i] = models_r

    with open('results/spectra/spectra_binned.pkl','wb') as f:
        pickle.dump(models_binned,f)

if __name__ == '__main__':
    compute_spectra()
    compute_statistics()
    







