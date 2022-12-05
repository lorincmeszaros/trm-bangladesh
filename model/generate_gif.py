# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 11:16:49 2022

@author: lorinc
"""
import imageio.v2 as imageio
import os

#Generate gifs
#waterlogging
png_dir = r'p:\11208012-011-nabaripoma\Model\Python\results\real\waterlogging'
images = []
filenames = []
for file_name in sorted(os.listdir(png_dir)):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        filenames.append(file_path) 
        # images.append(imageio.imread(file_path))

# imageio.mimsave(r'p:\11208012-011-nabaripoma\Model\Python\results\real\waterlogging\waterlogging.gif', images, fps=2)
 
# # Make it pause at the end so that the viewers can ponder
# for _ in range(10):
#     images.append(imageio.imread(file_path))
      
# with imageio.get_writer(r'p:\11208012-011-nabaripoma\Model\Python\results\real\waterlogging\waterlogging.gif', mode='I') as writer:
#     for image in images:
#         writer.append_data(image)
        
with imageio.get_writer(r'p:\11208012-011-nabaripoma\Model\Python\results\real\waterlogging\waterlogging.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# #gross-income
# with imageio.get_writer(r'p:\11208012-011-nabaripoma\Model\Python\results\real\gross_income\gross_income.gif', mode='I') as writer:
#     for filename in filenames_gross_income:
#         image = imageio.imread(filename)
#         writer.append_data(image)