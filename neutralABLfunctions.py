
''' Neutral ABL functions from OpenFOAM guide '''

import numpy as np

def neutralABL(z,zref,Uref,z0,kvk,C1,C2,Cmu):
    ufric = Uref*kvk/np.log((zref+z0)/z0)
    k = ufric**2/np.sqrt(Cmu)*np.sqrt(C1*np.log((z+z0)/z0)+C2)
    eps = ufric**3/(kvk*(z+z0))*np.sqrt(C1*np.log((z+z0)/z0)+C2)
    omega = ufric/(kvk*np.sqrt(Cmu))*1/(z+z0)
    u = ufric/kvk*np.log((z+z0)/z0)
    return k, omega, ufric, eps, u