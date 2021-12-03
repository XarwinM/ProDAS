import torch
import numpy as np

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



if __name__ == "__main__":

    n_samples =64
    
    # The following creates a list of length n_samples
    # Each element in the list is a tuple (sample, meta_data)
    samples = model.sample(n=n_samples, distribution='default')
    
    dataset = torch.cat( [torch.tensor(s[0]).view(1, *s[0].shape) for s in samples] )
    dataset = torch.utils.data.TensorDataset(dataset)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
