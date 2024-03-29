
''' Create boundaryData from TurbSim.bts '''

import numpy as np
from pyFAST.input_output import TurbSimFile
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

### Choose Operations ###
PERIODIC = 1                            # create an extension of inflow in the y-direction (doubling y length)
                                        # by flipping/mirroring the original field
FLIP_of_FLIP = 1                        # quadrupled y length
### Mod ###
turb_sim_path = 'TurbSim.bts'
# turb_sim_path = 'data/TurbSim.bts'
NlogLawInterpolationPoints_W = 5      	# number of layers to add near wall
NlogLawInterpolationPoints_H = 5      	# number of layers to add in the upper part
boundaryNames = ['Inlet','Outlet']
xPatch = [-960, 3840]                   # x-coordinate of the patches
folder_name = 'constant/boundaryData'  	# boundaryData folder path

z_ref = 150			                    # [m] zHub
zHub = z_ref
u_ref = 7.5				                # [m/s] velocity at hub
dt = 0.5				                # [s] time step of the series
TI = 7                                  # [%] turbulence intensity
ke_thrsh = 1e-3                        # kinetic energy threshold
MOlength = 41.8                         # [m] TO CHECK #################################################################
PBLH = 1000                             # [m] Planetary Boundary Layer Height
z0 = 0.0005                             # [m] roughness
kvk = 0.41                              # von Karman constant
Cmu = 0.09                              # Cmu
C1 = 0                                  # C1
C2 = 1                                  # C2
lmax = MOlength                         # used for the first added layer in the upper part
#___________________________________________FUNCTIONS___________________________________________________________________
def interpCenter(matrix):
    prev_rows = matrix[matrix.shape[0] // 2 - 1: matrix.shape[0] // 2]
    next_rows = matrix[matrix.shape[0] // 2: matrix.shape[0] // 2 + 1]
    means = np.mean(np.vstack((prev_rows, next_rows)), axis=0)
    mid_index = matrix.shape[0] // 2
    new_matrix = np.insert(matrix, mid_index, means, axis=0)
    return new_matrix
    
def neutralABL(z,zref,Uref,z0,kvk,C1,C2,Cmu):
    ufric = Uref*kvk/np.log((zref+z0)/z0)
    k = ufric**2/np.sqrt(Cmu)*np.sqrt(C1*np.log((z+z0)/z0)+C2)
    eps = ufric**3/(kvk*(z+z0))*np.sqrt(C1*np.log((z+z0)/z0)+C2)
    omega = ufric/(kvk*np.sqrt(Cmu))*1/(z+z0)
    u = ufric/kvk*np.log((z+z0)/z0)
    return k, omega, ufric, eps, u
#____________________________________READ_TURBSIM_BTS___________________________________________________________________
ts = TurbSimFile(turb_sim_path)
NyPoints = len(ts['y'])
NzPoints = len(ts['z'])
time = ts['t']
deltaY = abs(ts['y'][0] - ts['y'][1])
Y = ts['y']
Ntime = len(ts['t'])
#_______________________________________CREATE_boundaryData_FOR_OF______________________________________________________
### create boundaryData directory ###
folder_path = os.path.join(os.getcwd(), folder_name)    # Define the path to the folder
if os.path.isdir(folder_path):      # Check if the folder exists
    print('The folder boundaryData already exists\n')
    print('...........deleting all files and folders in boundaryData...........\n')
    for filename in os.listdir(folder_path):    # If the folder exists, delete all the files in it
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e)+'\n')
else:
    os.makedirs(folder_path)            # If the folder does not exist, create it
    print('The folder boundaryData has been created\n')

### add layers to turbsim ###
z_W = np.linspace(z0, ts['z'][0], NlogLawInterpolationPoints_W)
z_H = np.linspace(max(ts['z'])+lmax,PBLH,NlogLawInterpolationPoints_H)
z = ts['z']
Z = np.concatenate((z_W[0:-1],z,z_H))
Z = np.insert(Z,0,0)    # first element is 0

### start filling boundaryData folder ###
count=0
for patch in boundaryNames:
    if not os.path.exists(folder_path+'/'+patch):
        os.makedirs(folder_path+'/'+patch)
        print('The folder ' + patch + ' has been created\n')
    else:
        print('The folder ' + patch + ' already exists\n')

    for t in range(0,Ntime):

        # create grid data for all vars
        UU = np.zeros((NyPoints, len(Z)))    # rows are y, columns are z
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
        for j in range(0,NyPoints):
            # homogeneous velocity profile for upper layers
            k, omega, ufric, eps, u = neutralABL(Z, ts['z'][-1], ts['u'][0,t,j,-1], z0, kvk, C1, C2, Cmu)
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
            z_log = Z[NlogLawInterpolationPoints_W:NzPoints+NlogLawInterpolationPoints_W]
            kRANS = k[NlogLawInterpolationPoints_W:NzPoints+NlogLawInterpolationPoints_W]
            U_log = u_ref * np.log(z_log/z0)/np.log(z_ref/z0)
            k_resolved = 2/3 * (TI/100 * U_log)**2
            KE[j, NlogLawInterpolationPoints_W:NzPoints+NlogLawInterpolationPoints_W] = kRANS - k_resolved
            # TKE threshold
            KE[KE <= ke_thrsh] = ke_thrsh

            # create cosDirs of the first turbsim layer (used for modelling u in wall layers)
            uMag_first_ts_layer = np.sqrt(ts['u'][0,t,j,0]**2+ts['u'][1,t,j,0]**2+ts['u'][2,t,j,0]**2)
            u_cosDir = ts['u'][0, t, j, 0] / uMag_first_ts_layer
            v_cosDir = ts['u'][1, t, j, 0] / uMag_first_ts_layer
            w_cosDir = ts['u'][2, t, j, 0] / uMag_first_ts_layer

            # wall levels for velocity components
            for i in range(0,NlogLawInterpolationPoints_W):
                kL, omegaL, ufricL, epsL, uMagL = neutralABL(Z[i], Z[NlogLawInterpolationPoints_W], uMag_first_ts_layer, z0, kvk, C1, C2, Cmu)
                uL = uMagL * u_cosDir
                vL = uMagL * v_cosDir
                wL = uMagL * w_cosDir
                UU[j, 1:NlogLawInterpolationPoints_W] = uL
                VV[j, 1:NlogLawInterpolationPoints_W] = vL
                WW[j, 1:NlogLawInterpolationPoints_W] = wL

        # mirroring/flipping for periodic conditions on y-dir (periodic direction)
        if PERIODIC == 1:
            if NyPoints % 2 == 0:   # if NyPoints are odd interpolate for the central value and place at y-lim
                if t==0 and count==0:
                    print('||||| Points in y-dir are even: interpolated values in y extremities! |||||\n')
                    Y = np.insert(Y, len(Y) // 2, 0.0)
                    added_y_pos = np.arange(0,len(Y)//2)*deltaY+Y[-1]+deltaY
                    added_y_neg = added_y_pos * -1
                    added_y_neg = added_y_neg[::-1]
                    Y = np.concatenate((added_y_neg, Y, added_y_pos))

                UU = interpCenter(UU)
                VV = interpCenter(VV)
                WW = interpCenter(WW)
                KE = interpCenter(KE)
                OMEGA = interpCenter(OMEGA)
                EPS = interpCenter(EPS)

                UU = np.concatenate((np.flip(UU[1:int(NyPoints / 2)+1], axis=0), UU,
                                     np.flip(UU[int(NyPoints / 2):NyPoints], axis=0)))
                VV = np.concatenate((np.flip(VV[1:int(NyPoints / 2) + 1], axis=0), VV,
                                     np.flip(VV[int(NyPoints / 2):NyPoints], axis=0)))
                WW = np.concatenate((np.flip(WW[1:int(NyPoints / 2) + 1], axis=0), WW,
                                     np.flip(WW[int(NyPoints / 2):NyPoints], axis=0)))
                EPS = np.concatenate((np.flip(EPS[1:int(NyPoints / 2) + 1], axis=0), EPS,
                                      np.flip(EPS[int(NyPoints / 2):NyPoints], axis=0)))
                KE = np.concatenate((np.flip(KE[1:int(NyPoints / 2) + 1], axis=0), KE,
                                         np.flip(KE[int(NyPoints / 2):NyPoints], axis=0)))
                OMEGA = np.concatenate((np.flip(OMEGA[1:int(NyPoints / 2) + 1], axis=0), OMEGA,
                                            np.flip(OMEGA[int(NyPoints / 2):NyPoints], axis=0)))

            else:
                if t == 0 and count==0:
                    print('||||| Points in y-dir are odd: same values in y extremities! |||||\n')
                    added_y_pos = np.arange(0, len(Y) // 2) * deltaY + Y[-1] + deltaY
                    added_y_neg = added_y_pos * -1
                    added_y_neg = added_y_neg[::-1]
                    Y = np.concatenate((added_y_neg, Y, added_y_pos))
                UU = np.concatenate((np.flip(UU[1:NyPoints//2], axis=0), UU,
                                    np.flip(UU[NyPoints//2-1:NyPoints-1], axis=0)))
                VV = np.concatenate((np.flip(VV[1:NyPoints // 2], axis=0), VV,
                                     np.flip(VV[NyPoints // 2 - 1:NyPoints - 1], axis=0)))
                WW = np.concatenate((np.flip(WW[1:NyPoints // 2], axis=0), WW,
                                     np.flip(WW[NyPoints // 2 - 1:NyPoints - 1], axis=0)))
                EPS = np.concatenate((np.flip(EPS[1:NyPoints // 2], axis=0), EPS,
                                     np.flip(EPS[NyPoints // 2 - 1:NyPoints - 1], axis=0)))
                KE = np.concatenate((np.flip(KE[1:NyPoints // 2], axis=0), KE,
                                     np.flip(KE[NyPoints // 2 - 1:NyPoints - 1], axis=0)))
                OMEGA = np.concatenate((np.flip(OMEGA[1:NyPoints // 2], axis=0), OMEGA,
                                     np.flip(OMEGA[NyPoints // 2 - 1:NyPoints - 1], axis=0)))
        if FLIP_of_FLIP == 1:
            if t==0 and count==0:
                added_y_pos = np.arange(0, len(Y) // 2) * deltaY + Y[-1] + deltaY
                added_y_neg = added_y_pos * -1
                added_y_neg = added_y_neg[::-1]
                Y = np.concatenate((added_y_neg, Y, added_y_pos))
            UU = np.concatenate((np.flip(UU[1:len(UU) // 2], axis=0), UU,
                                 np.flip(UU[len(UU) // 2 - 1:len(UU) - 1], axis=0)))
            VV = np.concatenate((np.flip(VV[1:len(VV) // 2], axis=0), VV,
                                 np.flip(VV[len(VV) // 2 - 1:len(VV) - 1], axis=0)))
            WW = np.concatenate((np.flip(WW[1:len(WW) // 2], axis=0), WW,
                                 np.flip(WW[len(WW) // 2 - 1:len(WW) - 1], axis=0)))
            EPS = np.concatenate((np.flip(EPS[1:len(EPS) // 2], axis=0), EPS,
                                  np.flip(EPS[len(EPS) // 2 - 1:len(EPS) - 1], axis=0)))
            KE = np.concatenate((np.flip(KE[1:len(KE) // 2], axis=0), KE,
                                 np.flip(KE[len(KE) // 2 - 1:len(KE) - 1], axis=0)))
            OMEGA = np.concatenate((np.flip(OMEGA[1:len(OMEGA) // 2], axis=0), OMEGA,
                                    np.flip(OMEGA[len(OMEGA) // 2 - 1:len(OMEGA) - 1], axis=0)))

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
    fid_0.write(f'{len(Y)*len(Z)} \n')
    fid_0.write('(\n')
    for i in range(0, len(Y)):
        for j in range(0, len(Z)):
            fid_0.write(f'({xPatch[count]:.15f} {Y[i]:.15f} {Z[j]-zHub:.15f})\n')
    fid_0.write(')\n')
    fid_0.write('// ************************************************************************* //')
    count=count+1
