from .perlin_noise import SimplexNoise

from multiprocessing import Pool
from glob import glob
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
from skimage import draw
from skimage import transform

IMSIZE = 64
ANTIALIASING = 8

def _load_textures():
    texture_fnames = glob('./textures/texture_??.png')
    return [plt.imread(f) for f in texture_fnames]

TEXTURES = _load_textures()

def render(seed = None,

           fg_color_1 = (1.0, 0.1, 0.1),
           fg_color_2 = (0.9, 0.9, 0.1),
           fg_texture = 1,
           fg_texture_offset = (0, 0),

           bg_color_1 = (0.1, 0.8, 0.1),
           bg_color_2 = (0.1, 0.1, 0.8),
           bg_texture = 0,
           bg_texture_offset = (0, 0),

           object_size = 0.3,
           object_shape = 'circle',
           object_position = (0.5, 0.5),
           object_rotation = 0.,

           texture_contrast = 0.5,
           texture_noise_scale = 0.02,
           texture_noise_strength = 0.5,

           hsv_noise_scale = 0.02,
           hsv_noise_channel_coherence = 8.,
           hsv_noise_strength = (0.05, 0.15, 0.20),

           gauss_noise_strength = 0.02

           ):

    # two different perlin noise sources
    texture_noise = SimplexNoise(period=1024, seed=seed)
    hsv_noise = SimplexNoise(period=1024, seed=(seed + 1 if seed else seed))

    im = np.zeros((IMSIZE, IMSIZE, 3))

    # for antialiasing, render the shapes at a larger size
    aa_size = ANTIALIASING * IMSIZE
    fg_mask = np.zeros((aa_size, aa_size))

    if object_shape == 'circle':
        fg_mask_indeces = draw.circle(object_position[0] * aa_size,
                                      object_position[1] * aa_size,
                                      0.5 * object_size * aa_size)
    else:

        if object_shape == 'square':
            vertices = np.array([ (-0.5, -0.5),
                                  (-0.5,  0.5),
                                  ( 0.5,  0.5),
                                  ( 0.5, -0.5) ])
            # make it so the area is the same as the circle
            # for the same 'object_size' argument
            vertices *= 0.886

        elif object_shape == 'triangle':

            vertices = np.array([ ( 0., np.sqrt(3)),
                                  (-2,  -np.sqrt(3)),
                                  ( 2,  -np.sqrt(3)) ])

            # make it so the area is the same as the circle
            # for the same 'object_size' argument
            vertices *= 0.25 * 1.347

        else:
            raise ValueError("Shape must be 'square', 'circle' or 'triangle'")

        alpha = 2. * np.pi * object_rotation
        rot_matrix = np.array([ (np.cos(alpha), -np.sin(alpha)),
                                (np.sin(alpha),  np.cos(alpha)) ])

        vertices = vertices @ rot_matrix.T
        vertices *= object_size
        vertices += np.array([object_position])
        vertices *= aa_size

        fg_mask_indeces = draw.polygon(vertices[:,0], vertices[:,1])

    fg_mask_indeces = np.array(fg_mask_indeces)

    # kick out the indexes that go over the image boundary
    valid_indexes = np.logical_and(fg_mask_indeces >= 0, fg_mask_indeces < aa_size)
    valid_indexes = np.logical_and(valid_indexes[0], valid_indexes[1])

    fg_mask_indeces = fg_mask_indeces[:, valid_indexes]

    fg_mask[fg_mask_indeces[0], fg_mask_indeces[1]] = 1.
    fg_mask = transform.resize(fg_mask, (IMSIZE, IMSIZE), order=3)

    color_1 = np.array((bg_color_1, fg_color_1))
    color_2 = np.array((bg_color_2, fg_color_2))

    textures = np.stack([TEXTURES[bg_texture], TEXTURES[fg_texture]])
    texture_offset = (bg_texture_offset, fg_texture_offset)

    for i in tqdm(range(IMSIZE), ascii=True, ncols=100):
        for j in range(IMSIZE):

            pre_blend = np.zeros((2, 3))

            # the loop renders the background and the foreground separately
            for l in range(2):

                # skip this pixel for fore/background if it is not visible
                if abs(fg_mask[i, j] + l - 1) < 1e-8:
                    continue

                # find the texture pixel and add effects
                tex = textures[l][(i + texture_offset[l][0]) % 16, (j + texture_offset[l][1]) % 16]
                tex = texture_contrast * (tex - 0.5)
                tex *= 1 + texture_noise_strength * texture_noise.noise2(i * texture_noise_scale,
                                                                         j * texture_noise_scale)
                tex = max(0., min(tex + 0.5, 1.))

                rgb_vec = tex * color_1[l] + (1. - tex) * color_2[l]
                hsv_vec = colors.rgb_to_hsv(rgb_vec)

                # add the hsv perlin noise
                for k in range(3):
                    hsv_vec[k] += (hsv_noise_strength[k]
                                   * hsv_noise.noise3(i * hsv_noise_scale,
                                                      j * hsv_noise_scale,
                                                      float(k) / hsv_noise_channel_coherence + 10 * l))

                rgb_vec = colors.hsv_to_rgb(np.clip(hsv_vec, 0, 1))
                pre_blend[l] = rgb_vec

            # combine the fore- and background
            blend = pre_blend[1] * fg_mask[i,j] + pre_blend[0] * (1 - fg_mask[i,j])

            im[i, j] = blend

    im += gauss_noise_strength * np.random.randn(*im.shape)
    im = np.clip(im, 0, 1)
    return im

def render_wrap(arg):
    # just a wrapper for the vizualization
    # (has to be outside of __main__ clause due to multiprocessing)

    np.random.seed(None)
    random_colors = []
    for k in range(4):
        h = random.random()
        s = 0.5 + 0.5 * random.random()
        v = 0.3 + 0.7 * random.random()
        random_colors.append(colors.hsv_to_rgb((h, s, v)))

    return render(
                    bg_color_1 = random_colors[0],
                    bg_color_2 = random_colors[1],
                    fg_color_1 = random_colors[2],
                    fg_color_2 = random_colors[3],
                    object_position = 0.4 + 0.2 * np.random.rand(2),
                    object_size = 0.2 + 0.4 * random.random(),
                    object_shape = random.choice(['circle', 'square', 'triangle']),
                    object_rotation = random.random(),
                    bg_texture = np.random.randint(7),
                    fg_texture = np.random.randint(7),
                    bg_texture_offset = np.random.randint(15),
                    fg_texture_offset = np.random.randint(15),
            )

if __name__ == '__main__':

    mp_pool = Pool(16)
    images = mp_pool.map(render_wrap, range(16))

    for n in range(16):
        plt.subplot(4, 4, n + 1)
        plt.imshow(images[n])
        plt.axis('off')

    plt.tight_layout()
    plt.show()
