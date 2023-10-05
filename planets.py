class Star:
    radius : float # in Solar radii
    Teff : float # K
    metal : float # log10(M/H)
    kmag : float
    logg : float
    planets : dict # dictionary of planet objects

    def __init__(self, radius, Teff, metal, kmag, logg, planets):
        self.radius = radius
        self.Teff = Teff
        self.metal = metal
        self.kmag = kmag
        self.logg = logg
        self.planets = planets
        
class Planet:
    radius : float # in Jupiter radii
    mass : float # in Jupiter masses
    Teq : float # Equilibrium T in K
    transit_duration : float # in seconds
    a: float # semi-major axis in AU
    stellar_constant: float # W/m^2
    
    def __init__(self, radius, mass, Teq, transit_duration, a, stellar_constant):
        self.radius = radius
        self.mass = mass
        self.Teq = Teq
        self.transit_duration = transit_duration
        self.a = a
        self.stellar_constant = stellar_constant

k2_18b = Planet(
    radius=0.211, # Exo.Mast
    mass=0.0281, # Exo.Mast
    Teq=278.7, # Matches stellar constant
    transit_duration=2.682*60*60, # Exo.Mast
    a=0.143, # Exo.Mast
    stellar_constant=1368.0 # Benneke et al. (2019)
)

k2_18 = Star(
    radius=0.41, # Exo.Mast
    Teff=3457.0, # Exo.Mast
    metal=0.12, # Exo.Mast
    kmag=8.9, # Exo.Mast
    logg=4.79, # Exo.Mast
    planets={'b':k2_18b}
)








