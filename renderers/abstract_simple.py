import numpy as np
from skimage import draw
import matplotlib.pyplot as plt
import random

import pdb

class AbstractSimple:
    """
        Class that allows to sample images with varying number of objects;
        Sampling from in-distribuion and out-of-distribution is possible

    """
    def __init__(self):

        self.colors_background = np.array([ [0.55, 0.75, 0.95],
                        [0.80, 0.85, 0.90],
                        [0.10, 0.14, 0.35],
                        [0.65, 0.65, 0.65] ])

        self.colors_texture =  np.array([ [0.3, 0.3, 0.3],
                           [0.5, 0.5, 0.5],
                           [0.15, 0.45, 0.05],
                           [0.32, 0.25, 0.18] ])

        self.img_size = 64
        self.object_size = 8
        self.scale_max = 2

    def rotation(self, r, c, center=(0,0), rotation=0):
        """
            Rotates vertices of object around center=center with rotation=rotation
        """

        theta = np.radians(rotation)
        x1, s = np.cos(theta), np.sin(theta)
        R = np.array(((x1, -s), (s, x1)))

        for i in range(r.shape[0]):
            r[i] -= center[0]
            c[i] -= center[1]

            new = np.matmul(R, np.array([r[i],c[i]]) )
            r[i] = round(new[0])
            c[i] = round(new[1])

            r[i] += center[0]
            c[i] += center[1]

        return r, c

    def eclipse(self, position=(15,15), scale=0, scale_2=0, rotation=0):
        """
            Generates Mask for Eclipse
        """
        rr, cc = draw.ellipse(position[0], position[1], (self.object_size*(1+scale))/2, (self.object_size*(1+scale_2))/2, rotation=rotation)
        return rr, cc

    def circle(self,position=(15,15), scale=0):
        """
            Generates Mask For Circle
        """
        rr, cc = draw.circle(position[0], position[1], radius=(self.object_size*(1+scale))/2)#, shape=img.shape)
        return rr, cc

    def box(self, position=(2,2), rotation=0, scale=0):
        """
            Generates Mask for Rectangular/Box
        """
        length = self.object_size
        top_l = np.array([position[0] - ((1+scale)*length)//2, position[1]+ ((1+scale)*length)//2])
        top_r = np.array([position[0] + ((1+scale)*length)//2, position[1] + ((1+scale)*length)//2] )

        bottom_l = np.array([position[0] - ((1+scale)*length)//2, position[1] - ((1+scale)*length)//2])
        bottom_r = np.array([position[0] + ((1+scale)*length)//2, position[1] - ((1+scale)*length)//2])

        r = np.array([top_l[0], top_r[0], bottom_r[0], bottom_l[0]])
        c = np.array([top_l[1], top_r[1], bottom_r[1], bottom_l[1]])

        r, c = self.rotation(r, c, center=position, rotation=rotation)

        rr, cc = draw.polygon(r, c)
        return rr, cc

    def triangle(self, position=(2,2), rotation=0, scale=0):
        """
            Generates Mask for Triangular
        """

        length = self.object_size
        top = np.array([ position[0], position[1] + ((1+scale)*5)//2] )
        bottom_l = np.array([position[0] - ((1+scale)*length)//2, position[1] - ((1+scale)*length)//2])
        bottom_r = np.array([position[0] + ((1+scale)*length)//2, position[1] - ((1+scale)*length)//2])

        r = np.array([top[0], bottom_r[0], bottom_l[0]])
        c = np.array([top[1], bottom_r[1], bottom_l[1]])

        r, c = self.rotation(r, c, center=position, rotation=rotation)

        rr, cc = draw.polygon(r, c)
        return rr, cc

    def generate_abstract_position(self, number=2):#, obj_form=0, min_distance=15):
        """
            Draws from pre-defined positions; Positions are the positions of the objects
            Positions are middle-points of the four quarters of the image

            number: Number of positions to draws
        """

        positions = [ (16,16), (16,48), (48,48), (48,16)  ]

        out = []
        for a in random.sample(positions, k=number):
            out.append(np.array(a))
        return out

    def generate_objects(self,
            number=2,
            obj_text=0,
            obj_type=0,
            rotation_range=(0,360),
            scale_range=(0.2,1)):
        """
            Generate Meta-Data for objects for one image (in distribution);
            Sampling of positions as in self.generate_abstract_position(...)

            number: Number of objects to generate
            obj_text: Texture of objects; range: [0,1,2,3]
            obj_type: Type of objects; 0: Triangle, 1: Box, 2: Circle, 3: Eclipse (the same for all objects)
            rotation_range: Range of rotations; objects rotation is uniformly sampled from this range (for each object separately)
            scale_range: Scaling range; the scale of each object is uniformly sampled from this range (for each object separately)

            return: Meta-data of objects (positions, object texture, scale, objecte type and rotation)
        """

        positions = self.generate_abstract_position(number=number)
        objects = []
        for p in positions:
            objects.append({'position':p})

        for obj in objects:
            obj['obj_text'] = obj_text
            obj['scale'] = self.scale_max * ((scale_range[1]-scale_range[0])*np.random.rand(1)[0]+scale_range[0])

            ### Additional eclipse parameter
            obj['scale_eclipse'] = self.scale_max * ((scale_range[1]-scale_range[0])*np.random.rand(1)[0]+scale_range[0])

            ### Ensures that in case obj_type=Eclipse, really a eclipse is generated
            ### Check wheter 0.2 is enough??
            while (obj['scale_eclipse'] -obj['scale'])**2 < 0.2:
                obj['scale_eclipse'] = self.scale_max*np.random.rand(1)[0]

            obj['type'] = obj_type
            obj['rotation'] = np.random.randint(rotation_range[0], rotation_range[1])
        return objects


    def generate_instance(self,
            background_id=0,
            objects=[]):
        """ Generates Image from meta-information about object in images

            background_id: Defines background texture and is in [0,1,2,3]
            objects: list of objects; each element defines one object via a dictionary that contains values of rotation, position, scale and object texture

            output: Image of objects; Numpy array of shape (self.img_size, self.img_size, 3)

        """

        img = np.ones((self.img_size, self.img_size, 3)) * self.colors_background[background_id % 4]

        for e, obj in enumerate(objects):

            if obj['type'] == 0:
                rr, cc =  self.triangle(position=obj['position'], scale=obj['scale'], rotation=obj['rotation'])
            elif obj['type'] == 1:
                rr, cc =  self.box(position=obj['position'], scale=obj['scale'], rotation=obj['rotation'])
            elif obj['type'] == 2:
                rr, cc =  self.circle(position=obj['position'], scale=obj['scale'])
            elif obj['type'] == 3:
                rr, cc =  self.eclipse(position=obj['position'], scale=obj['scale'], rotation=obj['rotation'])

            img[rr, cc] = self.colors_texture[obj['obj_text'] % 4]

        ### Noise could be added to the image
        #img +=  1e-1 *  np.random.randn(self.img_size, self.img_size, 3)
        return img

    def sample_custom(self,
            background_id=0,
            number=1,
            obj_text=0,
            obj_type=0,
            rotation_range=(0,360),
            scale_range=(0.2,1)):
        """ Allows cusomized sampling due to different parameters and ranges
        """

        objects = self.generate_objects(number=number, obj_text=obj_text, obj_type=obj_type, rotation_range=rotation_range, scale_range=scale_range)

        return self.generate_instance(background_id=background_id, objects=objects)

    def sample_id(self):
        """ Sample one in distribution images;
            Pre-defined distribution on latent factors that generate

        """
        background_id = np.random.randint(3)
        number = np.random.randint(3)+1
        obj_text = np.random.randint(3)
        obj_text = np.random.randint(3)
        obj_type = random.choice([0,2])
        rotation_range = (0,360)
        scale_range = (0.2, 1)

        out = self.sample_custom(background_id=background_id,
                number=number,
                obj_text=obj_text,
                obj_type=obj_type,
                rotation_range=rotation_range,
                scale_range=scale_range)

        return out

    def sample_od(self,
            level_0=False,
            level_1=False,
            level_2=False):
        """ Suggestion for sampling out-of-distribution (OOD) images; level_ specifies which type of OOD should be generated

            level_0: Texture sampled out of distr. if True, otherwise in distr. (background and object texture)
            level_1: Object type sampled out of distr. if True; otherwise in distr.
            level_2: Number of objects out of distr if True; otherwise in distr. (counting)

            output: generated image due to specifications (levels)
        """

        #### Level of texture
        if level_0 == True:
            background_id = 3
            texture_range =[3]
        else:
            background_id = np.random.randint(3)
            texture_range = [0,1,2]

        obj_text=random.choice(texture_range)

        #### Level of multiple objects (counting)
        if level_2 == True:
            number = 4
        else:
            number = np.random.randint(2)+1

        #### Level of one object; type of object
        if level_1 == True:
            obj_type = random.choice([1,3])
        else:
            obj_type = random.choice([0,2])
        objects = self.generate_objects(number=number, obj_text=obj_text,  obj_type=obj_type)

        return self.generate_instance(background_id=background_id, objects=objects)

if __name__ == "__main__":
    obj = AbstractSimple()

    for i in range(16):
        im = obj.sample_id()

        ### Generate image which is on all levels out of distr.
        #im = obj.sample_od(level_0=True, level_1=True, level_2=True)

        plt.subplot(4, 4, i+1)
        plt.imshow(im,  cmap='Greys_r',  interpolation='nearest')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig('SampleInDistr.pdf')
    plt.show()

