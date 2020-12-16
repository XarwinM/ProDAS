import pprint

import numpy as np
import matplotlib.pyplot as plt

from latent_model import BaseGenerator
import latent_model.distributions as ds
from renderers.dspritesPP import render

c1 = np.array((66,  221, 237), dtype=np.float) / 255.
c2 = np.array((82,  78,  249), dtype=np.float) / 255.

c3 = np.array((249,  78, 206), dtype=np.float) / 255.
c4 = np.array((255, 154,  20), dtype=np.float) / 255.

model = BaseGenerator(render,

                      fg_color_1      = ds.ContinuousNormal(c1, 0.02),
                      fg_color_2      = ds.ContinuousNormal(c2, 0.02),
                      fg_texture      = ds.DiscreteChoice([0,1]),

                      bg_color_1      = ds.ContinuousNormal(c3, 0.02),
                      bg_color_2      = ds.ContinuousNormal(c4, 0.02),
                      bg_texture      = ds.DiscreteChoice([2,4]),

                      object_size     = ds.ContinuousUniform(0.15, 0.15),
                      object_shape    = ds.DiscreteChoice(['circle', 'square']),
                      object_position = ds.ContinuousUniform((0.65, 0.65), (0.2, 0.2)),
                      object_rotation = ds.ContinuousUniform(0., 1.)
                      )


model.set_distribution('in_distribution', inherit_from='default')

model.set_distribution('ood_color', inherit_from='default',
                       fg_color_1 = ds.ContinuousNormal(c3, 0.02),
                       fg_color_2 = ds.ContinuousNormal(c4, 0.02),
                       bg_color_1 = ds.ContinuousNormal(c1, 0.02),
                       bg_color_2 = ds.ContinuousNormal(c2, 0.02)
                       )

model.set_distribution('ood_texture', inherit_from='default',
                       fg_texture      = ds.DiscreteChoice([6]),
                       bg_texture      = ds.DiscreteChoice([7])
                       )

model.set_distribution('ood_shape', inherit_from='default',
                       object_shape    = ds.DiscreteChoice(['triangle'])
                       )

model.set_distribution('ood_position', inherit_from='default',
                       object_position = ds.ContinuousNormal((0.25, 0.25), 0.02)
                       )


N_preview_samples = 5

for dist in model.distributions:

    if dist == 'default':
        continue

    print(f'\nSampling from {dist}')
    samples = model.sample(n=N_preview_samples, distribution=dist)

    plt.figure(figsize=(N_preview_samples*2, 2.5))
    plt.gcf().suptitle(dist)

    for i, (s,p) in enumerate(samples):
        plt.subplot(1, N_preview_samples, i+1)
        plt.imshow(s)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'./example_figures/{dist}.jpg', dpi=120)

plt.show()
