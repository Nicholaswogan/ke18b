import numpy as np
from scipy import constants as const
import cantera as ct
from scipy import integrate
from photochem.utils._format import FormatSettings_main, MyDumper, Loader, yaml

def composition_from_metalicity(M_H_metalicity):

    # composition of the Sun (mol/mol)
    # From Table 8 in Lodders et al. (2009), "Abundances of the elements in the solar system"
    sun_mol = {
        'H': 0.921514888949834,
        'He': 0.07749066995740882,
        'O': 0.0004946606569284939,
        'C': 0.0002300986287941852,
        'N': 6.278165064228584e-05,
        'Si': 3.131024001891741e-05,
        'Mg': 3.101174053726646e-05,
        'Ne': 0.00010582900979606936,
        'Fe': 2.6994013922759826e-05,
        'S': 1.1755152117252983e-05
    }

    # Separate metals from non-metals
    metals = ['O','C','N','Si','Mg','Ne','Fe','S']
    non_metals = ['H','He']

    # Add up all metals
    mol_metals = 0.0
    for sp in metals:
        mol_metals += sun_mol[sp]

    # Compute the mol of H, metal and He of body
    mol_H_body = (10.0**M_H_metalicity * (mol_metals/sun_mol['H']) + 1.0 + sun_mol['He']/sun_mol['H'])**(-1.0)
    mol_metal_body = 10.0**M_H_metalicity * (mol_metals/sun_mol['H'])*mol_H_body
    mol_He_body = (sun_mol['He']/sun_mol['H'])*mol_H_body
    
    # Check everything worked out
    assert np.isclose((mol_metal_body/mol_H_body)/(mol_metals/sun_mol['H']), 10.0**M_H_metalicity)
    
    # Get metal composition
    metal_fractions = {}
    for sp in metals:
        metal_fractions[sp] = sun_mol[sp]/mol_metals

    # compute composition of the body
    mol_body = {}
    mol_body['H'] = mol_H_body
    mol_body['He'] = mol_He_body
    for sp in metals:
        mol_body[sp] = mol_metal_body*metal_fractions[sp]

    return mol_body

def composition_from_metalicity_for_atoms(atoms, M_H_metalicity):
    mol_body = composition_from_metalicity(M_H_metalicity)

    mol_tot = 0.0
    for atom in atoms:
        if atom in mol_body:
            mol_tot += mol_body[atom]

    mol_out = {}
    for atom in atoms:
        if atom in mol_body: 
            mol_out[atom] = mol_body[atom]/mol_tot
        else:
            mol_out[atom] = 0.0
    return mol_out

def equilibrium_temperature(stellar_radiation, bond_albedo):
    T_eq = ((stellar_radiation*(1.0 - bond_albedo))/(4.0*const.sigma))**(0.25)
    return T_eq 

class TempPress:

    def __init__(self, P, T):
        self.log10P = np.log10(P)[::-1] # P in dynes/cm^2
        self.T = T[::-1] # T in K

    def temperature(self, P):
        return np.interp(np.log10(P), self.log10P, self.T)
    
def gravity(radius, mass, z):
    "CGS units"
    G_grav = const.G
    grav = G_grav * (mass/1.0e3) / ((radius + z)/1.0e2)**2.0
    grav = grav*1.0e2 # convert to cgs
    return grav # cm/s^2

def rhs_alt(P, u, mubar, radius, mass, pt):

    # Currently altitude in cm
    z = u[0]

    # compute gravity
    grav = gravity(radius, mass, z)

    # interpolate P-T profile
    T = pt.temperature(P)

    k_boltz = const.Boltzmann*1e7
    dz_dP = -(k_boltz*T*const.Avogadro)/(mubar*grav*P)

    return np.array([dz_dP])

def chemical_equilibrium_PT(P, T, ct_file, atoms, M_H_metalicity):
    '''Given a P-T profile and metalicity, this function computes chemical
    chemical equilibrium for the entire atmospheric column. CGS units.
    '''

    comp = composition_from_metalicity_for_atoms(atoms, M_H_metalicity)
    gas = ct.Solution(ct_file)

    mubar = np.empty(P.shape[0])
    equi = {}
    for i,sp in enumerate(gas.species_names):
        equi[sp] = np.empty(P.shape[0])
    for i in range(P.shape[0]):
        gas.TPX = T[i],P[i]/10,comp
        gas.equilibrate('TP')
        mubar[i] = gas.mean_molecular_weight
        for j,sp in enumerate(gas.species_names):
            equi[sp][i] = gas.X[j]

    surf = {}
    for i,sp in enumerate(gas.species_names):
        surf[sp] = equi[sp][0]

    return equi, surf, mubar

def altitude_profile_PT(P, T, radius, mass, mubar):
    '''Computes altitude given P-T. CGS units.
    '''
    pt = TempPress(P, T)
    args = (mubar, radius, mass, pt)
    z0 = 0.0
    
    out = integrate.solve_ivp(rhs_alt, [P[0], P[-1]], np.array([z0]), t_eval=P, args=args, rtol=1e-5)

    z = out.y[0]

    return z

def write_atmosphere_file(filename, alt, press, den, temp, eddy, mix):

    fmt = '{:25}'
    with open(filename, 'w') as f:
        f.write(fmt.format('alt'))
        f.write(fmt.format('press'))
        f.write(fmt.format('den'))
        f.write(fmt.format('temp'))
        f.write(fmt.format('eddy'))

        for key in mix:
            f.write(fmt.format(key))
        
        f.write('\n')

        for i in range(press.shape[0]):
            f.write(fmt.format('%e'%alt[i]))
            f.write(fmt.format('%e'%press[i]))
            f.write(fmt.format('%e'%den[i]))
            f.write(fmt.format('%e'%temp[i]))
            f.write(fmt.format('%e'%eddy[i]))

            for key in mix:
                f.write(fmt.format('%e'%mix[key][i]))

            f.write('\n')

def surf_boundary_conditions(surf, min_mix, sp_to_exclude):
    bc_list = []
    for sp in surf:
        if surf[sp] > min_mix and sp not in sp_to_exclude:
            lb = {"type": "mix", "mix": float(surf[sp])}
            ub = {"type": "veff", "veff": 0.0}
            entry = {}
            entry['name'] = sp
            entry['lower-boundary'] = lb
            entry['upper-boundary'] = ub
            bc_list.append(entry)
    out = {}
    out['boundary-conditions'] = bc_list

    return out

def write_settings_file(filename, surf, min_mix, sp_to_exclude, top, P_surf, planet_mass, planet_radius):
    template_settings_file = \
    """
atmosphere-grid:
  bottom: 0.0
  top: 5.352689e+07
  number-of-layers: 100

photolysis-grid:
  regular-grid: true
  lower-wavelength: 92.5
  upper-wavelength: 855.0
  number-of-bins: 200

planet:
  background-gas: H2
  surface-pressure: 0.1
  planet-mass: 5.333730118514301e+28 # grams
  planet-radius: 1508481200.0 # cm
  surface-albedo: 0.0
  solar-zenith-angle: 60.0
  hydrogen-escape:
    type: none
  default-gas-lower-boundary: Moses
  water:
    fix-water-in-troposphere: false
    relative-humidity: 1.0
    gas-rainout: false
    rainfall-rate: 1.0
    tropopause-altitude: 0.0
    water-condensation: true
    condensation-rate: {A: 1.0e-8, rhc: 1, rh0: 1.05}
    """
    settings = yaml.safe_load(template_settings_file)

    settings['atmosphere-grid']['top'] = float(top)
    settings['planet']['surface-pressure'] = float(P_surf/1e6)
    settings['planet']['planet-mass'] = float(planet_mass)
    settings['planet']['planet-radius'] = float(planet_radius)

    # boundary conditions
    bc = surf_boundary_conditions(surf, min_mix, sp_to_exclude)
    settings['boundary-conditions'] = bc['boundary-conditions']

    out = FormatSettings_main(settings)
    with open(filename,'w') as f:
        yaml.dump(out, f, Dumper=MyDumper ,sort_keys=False, width=70)

