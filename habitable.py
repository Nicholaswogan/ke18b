import numpy as np
from threadpoolctl import threadpool_limits
from photochem.clima import AdiabatClimate
from photochem import Atmosphere
from photochem.utils._format import FormatSettings_main, MyDumper, Loader, yaml
from pathos.multiprocessing import ProcessingPool as Pool
import utils

def eddy_profile_like_Earth(log10P, log10P_trop):
    """Generates an eddy diffusion profile like Earth's
    """
    slope = (3.6 - 5.4)/(-1 - (-4))
    eddy_upper = 5.6
    eddy_trop = 5.0
    eddy_cold_trap = 3.6

    eddy = np.zeros(len(log10P))

    inds = np.where(log10P > log10P_trop)
    eddy[inds] = eddy_trop

    ind = np.max(inds) + 1

    x = log10P[ind]
    y = eddy_cold_trap
    b = y - slope*x

    for i in range(ind,len(log10P)):
        eddy[i] = slope*log10P[i] + b

    eddy[eddy>eddy_upper] = eddy_upper

    return 10.0**eddy

def make_settings(settings_in, settings_out, settings_dict):
    fil = open(settings_in,'r')
    data = yaml.load(fil,Loader=Loader)
    fil.close()

    data['atmosphere-grid']['top'] = float(settings_dict['top'])
    data['planet']['surface-pressure'] = float(settings_dict['surface-pressure'])
    data['planet']['water']['tropopause-altitude'] = float(settings_dict['tropopause-altitude'])
    for bc in settings_dict['boundary-conditions']:
        data['boundary-conditions'].append(bc)

    data = FormatSettings_main(data)
    
    fil = open(settings_out,'w')
    yaml.dump(data,fil,Dumper=MyDumper,sort_keys=False,width=70)
    fil.close()

def couple2photochem(c, settings_in, settings_out, atmosphere_out, eddy, extra_bcs):

    if isinstance(eddy, (int, float)) and not isinstance(eddy, bool):
        eddy_ = np.ones(c.z.shape[0])*eddy # eddy is just a scalar
    elif eddy == "ModernEarth":
        ind = (c.T-c.T_trop==0).argmax()
        log10P_trop = np.log10(c.P[ind]/1e6) # trop pressure in log10 bars
        log10P = np.log10(c.P.copy()/1e6) # pressure in the atmosphere in bars
        # compute eddy diffusion that is like Earth's
        eddy_ = eddy_profile_like_Earth(log10P, log10P_trop) 

    c.out2atmosphere_txt(atmosphere_out, eddy_, overwrite=True)

    ind = (c.T-c.T_trop==0).argmax()
    trop_alt = c.z[ind]

    settings = {}
    settings['top'] = float(c.z[-1])
    settings['surface-pressure'] = float(c.P_surf/1e6)
    settings['tropopause-altitude'] = float(trop_alt)
    settings['boundary-conditions'] = extra_bcs

    make_settings(settings_in, settings_out, settings)

def make_picaso_input_habitable(outfile):
    pc = Atmosphere('input/zahnle_earth_new.yaml',\
                outfile+'_settings.yaml',\
                "input/k2_18b_stellar_flux.txt",\
                outfile+'_atmosphere_c.txt')
    mix = {}
    mix['press'] = pc.wrk.pressure
    mix['temp'] = pc.var.temperature
    for i,sp in enumerate(pc.dat.species_names[pc.dat.np:-2]):
        ind = pc.dat.species_names.index(sp)
        tmp = pc.wrk.densities[ind,:]/pc.wrk.density
        mix[sp] = tmp
    species = pc.dat.species_names[pc.dat.np:-2]
    utils.write_picaso_atmosphere(mix, outfile+'_picaso.pt', species)

def climate_and_photochem(settings_in, outfile, eddy, extra_bcs,
                          mix, P_surf, T_trop, T_guess):
    settings_out = outfile+"_settings.yaml"
    atmosphere_out = outfile+"_atmosphere.txt"

    # Do climate calculation
    c = AdiabatClimate('input/habitable/species_climate.yaml',
                       'input/habitable/settings_climate_scale=0.7.yaml',
                       'input/k2_18b_stellar_flux.txt')
    f_i = np.ones(len(c.species_names))*1.0e-10
    for key in mix:
        ind = c.species_names.index(key)
        f_i[ind] = mix[key]
    P_surf = P_surf
    P_i = f_i*P_surf
    bg_gas = 'H2'
    c.T_trop = T_trop
    c.solve_for_T_trop = False
    c.RH = np.ones(len(c.species_names))
    c.P_top = 1.0e-3
    T_surf = c.surface_temperature_bg_gas(P_i, P_surf, bg_gas, T_guess=T_guess)

    # Write files
    couple2photochem(c, settings_in, settings_out, atmosphere_out, eddy, extra_bcs)

    # Initialize photochemical model
    pc = Atmosphere('input/zahnle_earth_new.yaml',\
                    settings_out,\
                    "input/k2_18b_stellar_flux.txt",\
                    atmosphere_out)
    
    # Integrate to equilibrium
    pc.var.atol = 1e-25
    pc.var.rtol = 1e-3
    pc.initialize_stepper(pc.wrk.usol)
    tn = 0.0
    while tn < pc.var.equilibrium_time:
        tn = pc.step()
    
    # Write output file
    atmosphere_out_c = outfile+"_atmosphere_c.txt"
    pc.out2atmosphere_txt(atmosphere_out_c, overwrite=True)

    # Write picaso file
    make_picaso_input_habitable(outfile)

def default_params():
    params = {}
    params['settings_in'] = 'input/habitable/settings_habitable_template.yaml'
    params['outfile'] = None
    params['eddy'] = 5.0e5
    params['extra_bcs'] = []
    params['mix'] = {'H2O': 200.0, 'CO2': 1.0e-2, 'N2': 1.0e-2, 
                     'H2': 1.0, 'CH4': 1.0e-10, 'CO': 1.0e-10}
    params['P_surf'] = 1.0e6
    params['T_trop'] = 215.0
    params['T_guess'] = 350.0
    return params

def noCH4():
    params = default_params()
    params['outfile'] = 'results/habitable/noCH4'
    return params

def withCH4():
    params = default_params()
    params['outfile'] = 'results/habitable/withCH4'
    params['mix']['CH4'] = 1.0e-2

    bc = {}
    bc['name'] = 'CH4'
    bc['lower-boundary'] = {'type': 'mix', 'mix': 1.0e-2}
    bc['upper-boundary'] = {'type': 'veff', 'veff': 0.0}
    params['extra_bcs'].append(bc)
    return params

def withCH4_vdepCO():
    params = default_params()
    params['outfile'] = 'results/habitable/withCH4_vdepCO'
    params['mix']['CH4'] = 1.0e-2

    bc = {}
    bc['name'] = 'CH4'
    bc['lower-boundary'] = {'type': 'mix', 'mix': 1.0e-2}
    bc['upper-boundary'] = {'type': 'veff', 'veff': 0.0}
    params['extra_bcs'].append(bc)

    bc = {}
    bc['name'] = 'CO'
    bc['lower-boundary'] = {'type': 'vdep', 'vdep': 1.2e-4}
    bc['upper-boundary'] = {'type': 'veff', 'veff': 0.0}
    params['extra_bcs'].append(bc)

    return params

if __name__ == "__main__":
    threadpool_limits(limits=1)
    models = [
        noCH4,
        withCH4,
        withCH4_vdepCO
    ]
    def wrap(model):
        climate_and_photochem(**model())
    p = Pool(3)
    p.map(wrap, models)













