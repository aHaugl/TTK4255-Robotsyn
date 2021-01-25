# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:38:38 2021
Just some other image helping functions
@author: Andreas
"""

def ndg(img,rows,cols):
   mat = [[0 for x in range(cols)] for y in range(rows)]
   for x in range(cols):
       for y in range(rows):
           mat[x][y] = 0
           for x in range(cols):
               for y in range(rows):
                   val = img.getpixel((x, y))

                   mat[x][y] = val[2]*0.299+val[1]*0.587+val[0]*0.114
                   print(mat[x][y])
                   return mat
   print('mat ',ndg(img,rows,cols))
   
   
def normalized(down):

     norm=np.zeros((600,800,3),np.float32)
     norm_rgb=np.zeros((600,800,3),np.uint8)

     b=rgb[:,:,0]
     g=rgb[:,:,1]
     r=rgb[:,:,2]

     sum=b+g+r

     norm[:,:,0]=b/sum*255.0
     norm[:,:,1]=g/sum*255.0
     norm[:,:,2]=r/sum*255.0

     norm_rgb=cv2.convertScaleAbs(norm)
     return norm_rgb
