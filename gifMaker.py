
'''
    After SHOW_TURBSIM_DATA has been runned this script produce
    a .gif animation
'''

import imageio
import os
from natsort import natsorted

if not os.path.exists("results"):
    os.makedirs("results")

in_path = 'temp1/'
out_filename="animation"
out_path="results/"
in_filenames = natsorted(os.listdir(in_path))

with imageio.get_writer(out_path+out_filename+".gif", mode='I', duration='0.25') as writer:
    for in_filename in in_filenames:
        print(in_path+in_filename)
        image = imageio.imread(in_path + in_filename)
        writer.append_data(image)