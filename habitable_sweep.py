from photochemclima import PhotochemClima
import numpy as np
import pickle
from p_tqdm import p_map
import os
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)

def get_inputs():
    eddy_s = np.logspace(4.0,6.0,30)
    f_N2_s = np.logspace(-7.0,-2.0,30)

    inputs = []
    for i,e in enumerate(eddy_s):
        for j,f in enumerate(f_N2_s):
            inputs.append((i,j,e,f))

    return inputs

# Passed by scope
p = PhotochemClima('input/zahnle_earth_new.yaml',
                   'input/habitable/settings_habitable_template.yaml',
                   'input/k2_18b_stellar_flux.txt',
                   'input/habitable/atmosphere_init.txt',
                   'input/habitable/species_climate.yaml',
                   'input/habitable/settings_climate_scale=0.7.yaml')

def wrapper(x):
    eddy_i, f_N2_j, eddy, f_N2 = x
    RH = 1.0
    T_surf = 320
    output_file = 'results/habitable/sweep.pkl'

    mix = {'H2O': 200.0, 'CO2': 0.008, 'N2': f_N2}
    p.pc.set_lower_bc('N2',bc_type='mix',mix=f_N2)
    p.relative_humidity = RH
    p.constant_eddy = eddy
    res = p.find_equilibrium(T_surf, mix) # Compute photochemical equilibrium

    # save the result
    if not os.path.exists(output_file):
        with open(output_file, 'wb') as f:
            pass
    
    out = eddy_i, f_N2_j, eddy, f_N2, res
    with open(output_file,'ab') as f:
        pickle.dump(out, f)

if __name__ == "__main__":
    NUM_PROCESS = 40
    inputs = get_inputs()
    p_map(wrapper, inputs, num_cpus=NUM_PROCESS)
