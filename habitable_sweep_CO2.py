from photochemclima import PhotochemClima
import numpy as np
import pickle
from p_tqdm import p_umap
import os
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)

def get_inputs():
    eddy_s = np.logspace(4.0,6.0,20)
    f_N2_s = np.logspace(-6.0,-2.0,20)
    f_CO2_s = np.logspace(-4.0,np.log10(0.05),20)

    inputs = []
    for i,e in enumerate(eddy_s):
        for j,f in enumerate(f_N2_s):
            for k,c in enumerate(f_CO2_s):
                inputs.append((i,j,k,e,f,c))

    return inputs

# Passed by scope
p = PhotochemClima('input/zahnle_earth_new.yaml',
                   'input/habitable/settings_habitable_template.yaml',
                   'input/k2_18b_stellar_flux.txt',
                   'input/habitable/atmosphere_init.txt',
                   'input/habitable/species_climate.yaml',
                   'input/habitable/settings_climate_scale=0.7.yaml')

def wrapper(x):
    eddy_i, f_N2_j, f_CO2_k, eddy, f_N2, f_CO2 = x
    RH = 1.0
    T_surf = 320
    output_file = 'results/habitable/sweep_CO2.pkl'

    mix = {'H2O': 200.0, 'CO2': f_CO2, 'N2': f_N2}
    p.pc.set_lower_bc('N2',bc_type='mix',mix=f_N2)
    p.pc.set_lower_bc('CO2',bc_type='mix',mix=f_CO2)
    p.relative_humidity = RH
    p.constant_eddy = eddy
    res = p.find_equilibrium(T_surf, mix) # Compute photochemical equilibrium

    # save the result
    if not os.path.exists(output_file):
        with open(output_file, 'wb') as f:
            pass
    
    out = eddy_i, f_N2_j, f_CO2_k, eddy, f_N2, f_CO2, res
    with open(output_file,'ab') as f:
        pickle.dump(out, f)

if __name__ == "__main__":
    NUM_PROCESS = 40
    inputs = get_inputs()
    p_umap(wrapper, inputs, num_cpus=NUM_PROCESS)
