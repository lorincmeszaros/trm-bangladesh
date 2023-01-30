# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 11:16:49 2022

@author: lorinc
"""
import imageio.v2 as imageio
import os

png_dirs=[r'p:\11208012-011-nabaripoma\Model\Python\results\real\waterlogging\strategy1' ,
          r'p:\11208012-011-nabaripoma\Model\Python\results\real\waterlogging\strategy2']

gif_dirs=[r'p:\11208012-011-nabaripoma\Model\Python\results\real\waterlogging\strategy1\waterlogging_S1.gif' ,
          r'p:\11208012-011-nabaripoma\Model\Python\results\real\waterlogging\strategy2\waterlogging_S2.gif']

for st in [1, 2]:
    #Generate gifs
    #Waterlogging
    png_dir = png_dirs[st-1]
    images = []
    filenames = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            filenames.append(file_path) 
            # images.append(imageio.imread(file_path))
    
    print('All images listed for strategy' + str(st))
            
    with imageio.get_writer(gif_dirs[st-1], mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    print('All images read and merged into GIF for strategy' + str(st))
