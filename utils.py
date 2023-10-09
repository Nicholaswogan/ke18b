import numpy as np
from scipy import constants as const
import miepython

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

def write_picaso_atmosphere(mix, outfile, species):
    fmt = '{:25}'
    with open(outfile,'w') as f:
        f.write(fmt.format('pressure'))
        f.write(fmt.format('temperature'))
        for sp in species:
            f.write(fmt.format(sp))
        f.write('\n')
    
        P = mix['press'][::-1]/1e6
        T = mix['temp'][::-1]
    
        for i in range(P.shape[0]):
            f.write(fmt.format('%e'%P[i]))
            f.write(fmt.format('%e'%T[i]))
            for j,sp in enumerate(species):
                f.write(fmt.format('%e'%(mix[sp][P.shape[0]-i-1])))
            f.write('\n')

def residuals(data_y, err, expected_y):
    return (data_y - expected_y)/err

def chi_squared(data_y, err, expected_y):
    R = residuals(data_y, err, expected_y)
    return np.sum(R**2)

def reduced_chi_squared(data_y, err, expected_y, dof):
    chi2 = chi_squared(data_y, err, expected_y)
    return chi2/dof

def make_haze_opacity_file(pressure, haze_column, outfile):

    particle_radius_um = 0.1 # microns
    particle_radius_cm = particle_radius_um*(1/1e6)*(1e2/1) # cm 
    
    fil = open('data/khare_tholins.dat')
    lines = fil.readlines()
    fil.close()
    
    wavelength = [] # microns
    m_real = []
    m_imag = []
    for line in lines[13:]:
        tmp = line.split()
        wavelength.append(float(tmp[1]))
        m_real.append(float(tmp[2]))
        m_imag.append(float(tmp[3]))
        
    wavelength = np.array(wavelength) # this is micro meter
    m_real = np.array(m_real)
    m_imag = np.array(m_imag)
    
    radius = particle_radius_um
    x = 2 * np.pi * radius / wavelength
    m = m_real - 1j*m_imag
    qext, qsca, qback, g = miepython.mie(m, x)
    w0 = qsca/qext
    
    keys = ['pressure','wavenumber','opd','w0','g0']
    out = {}
    for key in keys:
        out[key] = []

    for j in range(haze_column.shape[0]):
        for i in range(qext.shape[0]):
            taup_1 = qext[i]*particle_radius_cm**2*haze_column[j]
    
            out['pressure'].append(pressure[j]/1e6)
            out['wavenumber'].append(1e4/wavelength[i])
            out['opd'].append(taup_1)
            out['w0'].append(w0[i])
            out['g0'].append(g[i])

    fmt = '{:20}'
    with open(outfile,'w') as f:
        for key in out:
            f.write(fmt.format(key))
        f.write('\n')
        for i in range(len(out['pressure'])):
            for key in out:
                f.write(fmt.format('%e'%(out[key][i])))
            f.write('\n')

def haze_production_rate(pc):
    res = {}
    
    pl = pc.production_and_loss('HCaer1',pc.wrk.usol)
    ind = pl.production_rx.index('C2H + C4H2 => HCaer1 + H')
    res['HCaer1_prod'] = pl.integrated_production[ind]
    
    pl = pc.production_and_loss('HCaer2',pc.wrk.usol)
    ind = pl.production_rx.index('H2CN + HCN => HCaer2')
    res['HCaer2_prod'] = pl.integrated_production[ind]
    
    pl = pc.production_and_loss('HCaer3',pc.wrk.usol)
    ind = pl.production_rx.index('C4H + HCCCN => HCaer3')
    res['HCaer3_prod'] = pl.integrated_production[ind]

    # molecules/cm^2/s * mol/molecules * g/mol = g/cm^2/s
    haze_prod = 0.0
    ind = pc.dat.species_names.index('HCaer1')
    haze_prod += (res['HCaer1_prod']/const.Avogadro)*pc.dat.species_mass[ind] 
    ind = pc.dat.species_names.index('HCaer2')
    haze_prod += (res['HCaer2_prod']/const.Avogadro)*pc.dat.species_mass[ind] 
    ind = pc.dat.species_names.index('HCaer3')
    haze_prod += (res['HCaer3_prod']/const.Avogadro)*pc.dat.species_mass[ind] 
    res['haze_prod'] = haze_prod
    return res