import numpy as np
from threadpoolctl import threadpool_limits
from pathos.multiprocessing import ProcessingPool as Pool
from photochemclima import PhotochemClima
import pickle
import utils

def make_picaso_input_habitable(p, outfile):
    pc = p.pc
    mix = {}
    mix['press'] = pc.wrk.pressure
    mix['temp'] = pc.var.temperature
    for i,sp in enumerate(pc.dat.species_names[pc.dat.np:-2]):
        ind = pc.dat.species_names.index(sp)
        tmp = pc.wrk.densities[ind,:]/pc.wrk.density
        mix[sp] = tmp
    species = pc.dat.species_names[pc.dat.np:-2]
    utils.write_picaso_atmosphere(mix, outfile+'_picaso.pt', species)

def run_model(outfile, T_surf, mix, vdep, eddy, T_trop, relative_humidity):

    p = PhotochemClima('input/zahnle_earth_new.yaml',
                   'input/habitable/settings_habitable_template.yaml',
                   'input/k2_18b_stellar_flux.txt',
                   'input/habitable/atmosphere_init.txt',
                   'input/habitable/species_climate.yaml',
                   'input/habitable/settings_climate_scale=0.7.yaml')
    p.pc.var.verbose = 1

    # Other variables
    p.constant_eddy = eddy
    for sp in mix:
        if sp == 'H2O' or sp == 'H2':
            continue
        else:
            p.pc.set_lower_bc(sp,bc_type='mix',mix=mix[sp])
    for sp in vdep:
        p.pc.set_lower_bc(sp,bc_type='vdep',vdep=vdep[sp])
    p.relative_humidity = relative_humidity
    p.c.T_trop = T_trop

    # photochemical equilibrium
    res = p.find_equilibrium(T_surf, mix)

    # Write output file
    atmosphere_out_c = outfile+"_atmosphere.pkl"
    with open(atmosphere_out_c,'wb') as f:
        pickle.dump(res,f)
    
    # Write picaso file
    make_picaso_input_habitable(p, outfile)
    
    # haze column density (particles/cm^2)
    ind = p.pc.dat.species_names.index('HCaer1')
    haze_density = p.pc.wrk.densities[ind,:]
    ind = p.pc.dat.species_names.index('HCaer2')
    haze_density += p.pc.wrk.densities[ind,:]
    ind = p.pc.dat.species_names.index('HCaer3')
    haze_density += p.pc.wrk.densities[ind,:]
    dz = p.pc.var.z[1] - p.pc.var.z[0]
    haze_column = haze_density*dz
    pressure = p.pc.wrk.pressure
    haze_file = outfile+'_haze.txt'
    utils.make_haze_opacity_file(pressure[:-1], haze_column[:-1], haze_file)

def default_params():
    params = {}
    params['outfile'] = None
    params['T_surf'] = 320.0
    params['mix'] = {'H2O': 200.0, 'CO2': 0.008, 'N2': 1.0e-2}
    params['vdep'] = {'CO': 0.0}
    params['eddy'] = 5.0e5
    params['T_trop'] = 215.0
    params['relative_humidity'] = 1.0
    return params

def model1a():
    params = default_params()
    params['outfile'] = 'results/habitable/model1a'
    params['mix'] = {'H2O': 200.0, 'CO2': 0.008, 'N2': 3.0e-3}
    params['eddy'] = 5.0e5
    return params

def model1b():
    params = default_params()
    params['outfile'] = 'results/habitable/model1b'
    params['mix'] = {'H2O': 200.0, 'CO2': 0.008, 'N2': 1.0e-6}
    params['eddy'] = 2.0e4
    return params

def model1c():
    params = default_params()
    params['outfile'] = 'results/habitable/model1c'
    params['mix'] = {'H2O': 200.0, 'CO2': 0.008, 'N2': 3.0e-3, 'CH4': 2.0e-2}
    params['vdep'] = {'CO': 1.2e-4}
    params['eddy'] = 5.0e5
    return params

if __name__ == "__main__":
    threadpool_limits(limits=1)
    models = [
        model1a,
        model1b,
        model1c
    ]
    def wrap(model):
        run_model(**model())
    p = Pool(3)
    p.map(wrap, models)













