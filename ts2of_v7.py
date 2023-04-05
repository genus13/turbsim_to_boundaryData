''' Create boundaryData from TurbSim.bts '''

import numpy as np
from pyFAST.input_output import TurbSimFile
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

### Choose Operations ###
PERIODIC_TRANSLATION = 1                # Translational periodicity
### Mod ###
turb_sim_path = 'TurbSim.bts'
NlogLawInterpolationPoints_W = 5  # number of layers to add near wall
NlogLawInterpolationPoints_H = 5  # number of layers to add in the upper part
boundaryNames = ['Inlet', 'Outlet']
xPatch = [-960, 3840]  # x-coordinate of the patches
folder_name = 'constant/boundaryData'  # boundaryData folder path

z_ref = 150  # [m] zHub
zHub = z_ref
u_ref = 7.5  # [m/s] velocity at hub
dt = 0.5  # [s] time step of the series
TI = 7  # [%] turbulence intensity
ke_thrsh = 1e-3  # kinetic energy threshold
MOlength = 41.8  # [m] TO CHECK #################################################################
PBLH = 1000  # [m] Planetary Boundary Layer Height
z0 = 0.0005  # [m] roughness
kvk = 0.41  # von Karman constant
Cmu = 0.09  # Cmu
C1 = 0  # C1
C2 = 1  # C2
lmax = MOlength  # used for the first added layer in the upper part
# ___________________________________________FUNCTIONS___________________________________________________________________

def neutralABL(z, zref, Uref, z0, kvk, C1, C2, Cmu):
    ufric = Uref * kvk / np.log((zref + z0) / z0)
    k = ufric ** 2 / np.sqrt(Cmu) * np.sqrt(C1 * np.log((z + z0) / z0) + C2)
    eps = ufric ** 3 / (kvk * (z + z0)) * np.sqrt(C1 * np.log((z + z0) / z0) + C2)
    omega = ufric / (kvk * np.sqrt(Cmu)) * 1 / (z + z0)
    u = ufric / kvk * np.log((z + z0) / z0)
    return k, omega, ufric, eps, u

# ____________________________________READ_TURBSIM_BTS___________________________________________________________________
ts = TurbSimFile(turb_sim_path)
NyPoints = len(ts['y'])
NzPoints = len(ts['z'])
time = ts['t']
deltaY = abs(ts['y'][0] - ts['y'][1])
Y = ts['y']
Ntime = len(ts['t'])
# _______________________________________CREATE_boundaryData_FOR_OF______________________________________________________
### create boundaryData directory ###
folder_path = os.path.join(os.getcwd(), folder_name)  # Define the path to the folder
if os.path.isdir(folder_path):  # Check if the folder exists
    print('The folder boundaryData already exists\n')
    print('...........deleting all files and folders in boundaryData...........\n')
    for filename in os.listdir(folder_path):  # If the folder exists, delete all the files in it
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e) + '\n')
else:
    os.makedirs(folder_path)  # If the folder does not exist, create it
    print('The folder boundaryData has been created\n')

### add layers to turbsim ###
z_W = np.linspace(z0, ts['z'][0], NlogLawInterpolationPoints_W)
z_H = np.linspace(max(ts['z']) + lmax, PBLH, NlogLawInterpolationPoints_H)
z = ts['z']
Z = np.concatenate((z_W[0:-1], z, z_H))
Z = np.insert(Z, 0, 0)  # first element is 0

### start filling boundaryData folder ###
count = 0
for patch in boundaryNames:
    if not os.path.exists(folder_path + '/' + patch):
        os.makedirs(folder_path + '/' + patch)
        print('The folder ' + patch + ' has been created\n')
    else:
        print('The folder ' + patch + ' already exists\n')

    for t in range(0, Ntime):

        # create grid data for all vars
        UU = np.zeros((NyPoints, len(Z)))  # rows are y, columns are z
        VV = np.zeros((NyPoints, len(Z)))
        WW = np.zeros((NyPoints, len(Z)))
        KE = np.zeros((NyPoints, len(Z)))
        EPS = np.zeros((NyPoints, len(Z)))
        OMEGA = np.zeros((NyPoints, len(Z)))

        # turbsim levels for velocity components
        UU[:, NlogLawInterpolationPoints_W:NlogLawInterpolationPoints_W + NzPoints] = ts['u'][0, t, :, :]
        VV[:, NlogLawInterpolationPoints_W:NlogLawInterpolationPoints_W + NzPoints] = ts['u'][1, t, :, :]
        WW[:, NlogLawInterpolationPoints_W:NlogLawInterpolationPoints_W + NzPoints] = ts['u'][2, t, :, :]

        # other levels (all for k, eps and omega and only upper for velocity components)
        end = NlogLawInterpolationPoints_W + NzPoints + NlogLawInterpolationPoints_H
        for j in range(0, NyPoints):
            # homogeneous velocity profile for upper layers
            k, omega, ufric, eps, u = neutralABL(Z, ts['z'][-1], ts['u'][0, t, j, -1], z0, kvk, C1, C2, Cmu)
            kH, omegaH, ufricH, epsH, uH = neutralABL(Z, z_ref, u_ref, z0, kvk, C1, C2, Cmu)
            UU[j, NzPoints + NlogLawInterpolationPoints_W:end] = uH[NzPoints + NlogLawInterpolationPoints_W:end]

            # non-homogeneous velocity profile for upper layers
            # k, omega, ufric, eps, u = neutralABL(Z, ts['z'][-1], ts['u'][0, t, j, -1], z0, kvk, C1, C2, Cmu)
            # UU[j, NzPoints + NlogLawInterpolationPoints_W:end] = u[NzPoints + NlogLawInterpolationPoints_W:end]

            VV[j, NzPoints + NlogLawInterpolationPoints_W:end] = 0
            WW[j, NzPoints + NlogLawInterpolationPoints_W:end] = 0
            EPS[j, 0:end] = eps
            OMEGA[j, 0:end] = omega

            ### TKE modelling ###
            # TKE upper layers
            # KE[j, NzPoints + NlogLawInterpolationPoints_W:end] = k[NzPoints + NlogLawInterpolationPoints_W:end]
            KE[j, NzPoints + NlogLawInterpolationPoints_W:end] = ke_thrsh
            # TKE wall layers
            KE[j, 0:NlogLawInterpolationPoints_W] = k[0:NlogLawInterpolationPoints_W]
            # TKE for turbsim layers
            z_log = Z[NlogLawInterpolationPoints_W:NzPoints + NlogLawInterpolationPoints_W]
            kRANS = k[NlogLawInterpolationPoints_W:NzPoints + NlogLawInterpolationPoints_W]
            U_log = u_ref * np.log(z_log / z0) / np.log(z_ref / z0)
            k_resolved = 2 / 3 * (TI / 100 * U_log) ** 2
            KE[j, NlogLawInterpolationPoints_W:NzPoints + NlogLawInterpolationPoints_W] = kRANS - k_resolved
            # TKE threshold
            KE[KE <= ke_thrsh] = ke_thrsh

            # create cosDirs of the first turbsim layer (used for modelling u in wall layers)
            uMag_first_ts_layer = np.sqrt(
                ts['u'][0, t, j, 0] ** 2 + ts['u'][1, t, j, 0] ** 2 + ts['u'][2, t, j, 0] ** 2)
            u_cosDir = ts['u'][0, t, j, 0] / uMag_first_ts_layer
            v_cosDir = ts['u'][1, t, j, 0] / uMag_first_ts_layer
            w_cosDir = ts['u'][2, t, j, 0] / uMag_first_ts_layer

            # wall levels for velocity components
            for i in range(0, NlogLawInterpolationPoints_W):
                kL, omegaL, ufricL, epsL, uMagL = neutralABL(Z[i], Z[NlogLawInterpolationPoints_W], uMag_first_ts_layer,
                                                             z0, kvk, C1, C2, Cmu)
                uL = uMagL * u_cosDir
                vL = uMagL * v_cosDir
                wL = uMagL * w_cosDir
                UU[j, 1:NlogLawInterpolationPoints_W] = uL
                VV[j, 1:NlogLawInterpolationPoints_W] = vL
                WW[j, 1:NlogLawInterpolationPoints_W] = wL

        # translational in both in pos and neg y-direction (triple initial y-span)
        if PERIODIC_TRANSLATION == 1:
            if NyPoints % 2 == 0:
                if t == 0 and count == 0:
                    print('||||| Points in y-dir are even! |||||\n')
                    add_pos_y = np.arange(0, len(Y) - 1) * deltaY + Y[-1] + deltaY
                    add_neg_y = add_pos_y * -1
                    add_neg_y = add_neg_y[::-1]
                    Y = np.concatenate((add_neg_y, Y, add_pos_y))
                UU = np.concatenate((UU[1:], UU, UU[:-1]))
                UU[-1] = UU[0]
                VV = np.concatenate((VV[1:], VV, VV[:-1]))
                VV[-1] = VV[0]
                WW = np.concatenate((WW[1:], WW, WW[:-1]))
                WW[-1] = WW[0]
                EPS = np.concatenate((EPS[1:], EPS, EPS[:-1]))
                EPS[-1] = EPS[0]
                KE = np.concatenate((KE[1:], KE, KE[:-1]))
                KE[-1] = KE[0]
                OMEGA = np.concatenate((OMEGA[1:], OMEGA, OMEGA[:-1]))
                OMEGA[-1] = OMEGA[0]
            else:
                if t == 0 and count == 0:
                    print('||||| Points in y-dir are odd! |||||\n')
                    add_pos_y = np.arange(0, len(Y)-1) * deltaY + Y[-1] + deltaY
                    add_neg_y = add_pos_y * -1
                    add_neg_y = add_neg_y[::-1]
                    Y = np.concatenate((add_neg_y, Y, add_pos_y))

                UU = np.concatenate((UU[1:],UU,UU[:-1]))
                UU[-1] = UU[0]
                VV = np.concatenate((VV[1:], VV, VV[:-1]))
                VV[-1] = VV[0]
                WW = np.concatenate((WW[1:], WW, WW[:-1]))
                WW[-1] = WW[0]
                EPS = np.concatenate((EPS[1:], EPS, EPS[:-1]))
                EPS[-1] = EPS[0]
                KE = np.concatenate((KE[1:], KE, KE[:-1]))
                KE[-1] = KE[0]
                OMEGA = np.concatenate((OMEGA[1:], OMEGA, OMEGA[:-1]))
                OMEGA[-1] = OMEGA[0]

        ### write boundaryData ###
        if not os.path.exists((folder_path + '/' + patch + '/' + str(ts['t'][t]))):
            os.makedirs(folder_path + '/' + patch + '/' + str(ts['t'][t]))

        ## write U,k,omega,epsilon ##

        # creating U,k,omega,epsilon files
        fid_U = open(folder_path + '/' + patch + '/' + str(ts['t'][t]) + '/' + 'U', "w")
        fid_k = open(folder_path + '/' + patch + '/' + str(ts['t'][t]) + '/' + 'k', "w")
        fid_eps = open(folder_path + '/' + patch + '/' + str(ts['t'][t]) + '/' + 'epsilon', "w")
        fid_omega = open(folder_path + '/' + patch + '/' + str(ts['t'][t]) + '/' + 'omega', "w")

        # header
        fid_U.write('(\n')
        fid_k.write("(\n")
        fid_eps.write("(\n")
        fid_omega.write("(\n")

        # looping over grid points
        for i in range(0, len(Y)):
            for j in range(0, len(Z)):
                fid_U.write(f'({UU[i, j]:.15f} {VV[i, j]:.15f} {WW[i, j]:.15f})\n')
                fid_k.write(f'{KE[i, j]:.15f}\n')
                fid_eps.write(f'{EPS[i, j]:.15f}\n')
                fid_omega.write(f'{OMEGA[i, j]:.15f}\n')

        # closing
        fid_U.write(')\n')
        fid_k.write(")\n")
        fid_eps.write(")\n")
        fid_omega.write(")\n")

        print('======', ts['t'][t], 'of', (Ntime - 1) * dt, 'seconds', '======\n')

    # creating 'points' file
    fid_0 = open(folder_path + '/' + patch + '/' + 'points', "w")
    fid_0.write('// Points \n')
    fid_0.write(f'{len(Y) * len(Z)} \n')
    fid_0.write('(\n')
    for i in range(0, len(Y)):
        for j in range(0, len(Z)):
            fid_0.write(f'({xPatch[count]:.15f} {Y[i]:.15f} {Z[j] - zHub:.15f})\n')
    fid_0.write(')\n')
    fid_0.write('// ************************************************************************* //')
    count = count + 1