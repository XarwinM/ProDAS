import numpy as np
import matplotlib.pyplot as plt

from latent_model import BaseGenerator
import latent_model.distributions as ds
from renderers.dspritesPP import render

import os

c1 = np.array((66,  221, 237), dtype=np.float64) / 255.
c2 = np.array((82,  78,  249), dtype=np.float64) / 255.

c3 = np.array((249,  78, 206), dtype=np.float64) / 255.
c4 = np.array((255, 154,  20), dtype=np.float64) / 255.

c5 = np.array((73, 255,  20), dtype=np.float) / 255.
c6 = np.array((124, 154,  124), dtype=np.float) / 255.

c7 = np.array((12, 255,  254), dtype=np.float) / 255.
c8 = np.array((12, 154,  200), dtype=np.float) / 255.


model = BaseGenerator(render,

                      fg_color_1      = ds.ContinuousNormal(c1, 0.02),
                      fg_color_2      = ds.ContinuousNormal(c2, 0.02),
                      fg_texture      = ds.DiscreteChoice([0, 1]),

                      bg_color_1      = ds.ContinuousNormal(c3, 0.02),
                      bg_color_2      = ds.ContinuousNormal(c4, 0.02),
                      bg_texture      = ds.DiscreteChoice([2, 4]),

                      object_size     = ds.ContinuousUniform(0.15, 0.15),
                      object_shape    = ds.DiscreteChoice(['circle', 'square']),
                      object_position = ds.ContinuousUniform((0.65, 0.65), (0.2, 0.2)),
                      object_rotation = ds.ContinuousUniform(0., 1.)
                      )

model.set_distribution('in_distribution', inherit_from='default')

model.set_distribution('ood_color_1', inherit_from='default',
                       fg_color_1 = ds.ContinuousNormal(c3, 0.02),
                       fg_color_2 = ds.ContinuousNormal(c4, 0.02),
                       bg_color_1 = ds.ContinuousNormal(c1, 0.02),
                       bg_color_2 = ds.ContinuousNormal(c2, 0.02)
                       )

model.set_distribution('ood_color_2', inherit_from='default',
                       fg_color_1 = ds.ContinuousNormal(c3, 0.02),
                       fg_color_2 = ds.ContinuousNormal(c4, 0.02),
                       bg_color_1 = ds.ContinuousNormal(c5, 0.02),
                       bg_color_2 = ds.ContinuousNormal(c6, 0.02)
                       )

model.set_distribution('ood_color_3', inherit_from='default',
                       fg_color_1 = ds.ContinuousNormal(c3, 0.02),
                       fg_color_2 = ds.ContinuousNormal(c4, 0.02),
                       bg_color_1 = ds.ContinuousNormal(c7, 0.02),
                       bg_color_2 = ds.ContinuousNormal(c8, 0.02)
                       )

model.set_distribution('ood_shape', inherit_from='default',
                       object_shape    = ds.DiscreteChoice(['triangle'])
                       )

model.set_distribution('ood_position', inherit_from='default',
                       object_position = ds.ContinuousNormal((0.25, 0.25), 0.02)
                       )

if __name__ == "__main__":

    N_samples = 1024
    np.random.seed(0)

    data_path = 'data/'

    object_dic = {'square': 0, 'circle': 1, 'triangle': 2}


    for dist in model.distributions:

        if dist == 'default':
            continue

        data_path_dist = data_path + dist

        try:
            os.mkdir(data_path_dist)
        except OSError:
            print ("Creation of the directory %s failed" % data_path_dist)

        print(f'\nSampling from {dist}')
        samples = model.sample(n=N_samples, distribution=dist)

        dataset_dist_data = []
        dataset_dist_label = []

        for i in range(N_samples):
            dataset_dist_label.append(  np.array([object_dic[samples[i][1]['object_shape']]]))
            dataset_dist_data.append(samples[i][0])

        np.save(data_path_dist+'/data.npy',  np.array(dataset_dist_data))
        np.save(data_path_dist+'/labels.npy',  np.array(dataset_dist_label))
