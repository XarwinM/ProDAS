import numpy as np
from skimage import draw
import matplotlib.pyplot as plt
import random


import pdb

class AbstractSimple:
    def __init__(self):

        self.colors_background = np.array([ [0.55, 0.75, 0.95],
                        [0.80, 0.85, 0.90],
                        [0.10, 0.14, 0.35],
                        [0.65, 0.65, 0.65] ])

        self.colors_texture = colors_ground = np.array([ [0.3, 0.3, 0.3],
                           [0.5, 0.5, 0.5],
                           [0.15, 0.45, 0.05],
                           [0.32, 0.25, 0.18] ])

        self.img_size = 64 

    def rotation(self, rr, cc):
        theta = np.radians(np.random.randint(360))
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        for i1 in range(rr.shape[0]):
            for i2 in range(rr.shape[0]): 
                new = np.matmul( np.array([rr[i1][i2],cc[i1][i2]]), R)
                rr[i1, i2] = new[0]
                cc[i1, i2] = new[1]

        return rr, cc 

    def rotation_2(self, r, c, center=(0,0)):


        theta = np.radians(np.random.randint(360))
        x1, s = np.cos(theta), np.sin(theta)
        R = np.array(((x1, -s), (s, x1)))

        for i in range(r.shape[0]):
            r[i] -= center[0]
            c[i] -= center[1]

            if i < 0:
                print(r[0], c[0], ' vs. ', np.matmul(R, np.array([r[i], c[i]]))) 
            new = np.matmul(R, np.array([r[i],c[i]]) )
            r[i] = round(new[0])
            c[i] = round(new[1])

            r[i] += center[0]
            c[i] += center[1]

        return r, c 

    def circle(self,img, position=(15,15), size=0 ):
        rr, cc = draw.circle(position[0], position[1], radius=int(5*(1+size)), shape=img.shape)
        return rr, cc
    
    def box (self,img, position=(2,2), size=0):
        extent = (int((1+size)*10), int((1+size)*10))
        rr, cc = draw.rectangle(position, extent=extent)#, shape=img.shape)

        rr, cc = self.rotation(rr, cc)
        
        return rr, cc

    def box_own(self, position=(2,2), rotation=0, size=0):
        top_l = np.array([position[0] - int((1+size)*5)//2, position[1]+ int((1+size)*5)//2])
        top_r = np.array([position[0] + int((1+size)*5)//2, position[1] + int((1+size)*5)//2] ) 

        bottom_l = np.array([position[0] - int((1+size)*5)//2, position[1] - int((1+size)*5)//2])
        bottom_r = np.array([position[0] + int((1+size)*5)//2, position[1] - int((1+size)*5)//2])

        r = np.array([top_l[0], top_r[0], bottom_r[0], bottom_l[0]])
        c = np.array([top_l[1], top_r[1], bottom_r[1], bottom_l[1]])

        r, c = self.rotation_2(r, c, center=position)

        #rr, cc = draw.polygon_perimeter(r, c)
        rr, cc = draw.polygon(r, c)
        #r = [19 ] 
        #c= [ 18 ]
        return rr, cc 

    def triangle_own(self, position=(2,2), rotation=0, size=0):

        length = 4
        top = np.array([ position[0], position[1] + int((1+size)*5)//2] ) 
        bottom_l = np.array([position[0] - int((1+size)*length)//2, position[1] - int((1+size)*length)//2])
        bottom_r = np.array([position[0] + int((1+size)*length)//2, position[1] - int((1+size)*length)//2])

        r = np.array([top[0], bottom_r[0], bottom_l[0]])
        c = np.array([top[1], bottom_r[1], bottom_l[1]])

        r, c = self.rotation_2(r, c, center=position)

        #rr, cc = draw.polygon_perimeter(r, c)
        rr, cc = draw.polygon(r, c)
        #r = [19 ] 
        #c= [ 18 ]
        return rr, cc 

    def generate_abstract_position(self, number=2, obj_form=0, min_distance=15):

        positions = []

        positions = [ (16,16), (16,48), (48,48), (48,16)  ]

        #for _ in range(number):
        #    (a,b) = np.random.randint(self.img_size-2*min_distance)+min_distance, np.random.randint(self.img_size-2*min_distance) + min_distance 
        liste = random.sample(positions, k=number)
        out = []
        for a in  liste:
            out.append(np.array(a)) 
        print(liste)
        return out 

    def generate_objects(self, number=2):

        positions = self.generate_abstract_position(number=number)
        objects = []
        for p in positions:
            objects.append({'position':p}) 

        for obj in objects:
            obj['obj_text'] = np.random.randint(4)
            obj['size'] = 5*np.random.rand(1)[0]
        return objects

    def generate_instance(self,
            number=2,
            background_id=0,
            obj_text=0,
            obj_form='box',
            obj_abstract=0):

        img = 0*np.ones((self.img_size, self.img_size, 3)) * self.colors_background[background_id % 4] + 0.05* np.random.randn(self.img_size, self.img_size, 3)

        #for obj in 
        objects = self.generate_objects(number=number) 

        for e, obj in enumerate(objects):
            #rr, cc =  self.box(img, position=obj['position'], size=obj['size'])
            #rr, cc =  self.circle(img, position=obj['position'], size=obj['size'])
            #rr, cc =  self.triangle(position=obj['position'], size=obj['size'])
            #rr, cc =  self.box_own(position=obj['position'], size=obj['size'])
            rr, cc =  self.triangle_own(position=obj['position'], size=obj['size'])
            #r = np.array([10,15, 25, 20])
            #c = np.array([20, 40, 40, 20])
            #rr, cc = draw.polygon(r, c)
            img[rr, cc] = 1#self.colors_texture[obj['obj_text'] % 4] 

        return img





if __name__ == "__main__":
    obj = AbstractSimple()

    for i in range(9):                                                                     
    #for i in range(1):                                                                     
        number = np.random.randint(3)+1
        im = obj.generate_instance(number=number) 

        plt.subplot(3, 3, i+1)
        plt.imshow(im,  cmap='Greys_r',  interpolation='nearest')                                        
        plt.xticks([])
        plt.yticks([])   

    print('Done')
    plt.tight_layout()              
    plt.savefig('test.pdf')
    #plt.show()

