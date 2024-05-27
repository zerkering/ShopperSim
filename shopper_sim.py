#!/usr/bin/env python
# coding: utf-8

# # Read me
# 
# Bigsimr is a library used in Julia. You cannot directly use it in Python.
# ref: https://pypi.org/project/bigsimr/
# 
# You have to install Julia and specify Julia's Path in Terminal first unless otherwises will error.
# To specify path, open your terminal and type: 
# export PATH="/Applications/Julia 1.4.2.app/Contents/Resources/julia/bin:$PATH"   and 
# sudo ln -s /Applications/Julia-1.4.2.app/Contents/Resources/julia/bin/julia /usr/local/bin/julia
# 
# Note: This depends on your Julia version
# 
# ref: https://stackoverflow.com/questions/62905587/julia-command-not-found-even-after-adding-to-path
# 

# In[1]:


# ! pip install simpy
# ! pip install git+https://github.com/SchisslerGroup/python-bigsimr.git
# ! pip install ipywidgets
# ! pip install ipython
# !jupyter nbextension enable --py widgetsnbextension --sys-prefix


# In[2]:


# from bigsimr import setup
# setup(compiled_modules=False)


# # Library

# In[3]:


#Julia library
from julia.api import Julia
jl = Julia(compiled_modules=False) # conda users -> set to False
from julia import Bigsimr as bs
from julia import Distributions as dist
#simpy
import simpy

#random and statistics
import random
import numpy as np
from scipy.stats import uniform,expon,lognorm,norm,rv_continuous,poisson,logistic
from scipy.special import logit,expit
from scipy.linalg import cholesky
import statistics as stat
import math
from collections import Counter

#ploting
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import animation
from ipywidgets import interact, Play
from matplotlib.cm import ScalarMappable

#widget
import ipywidgets as widgets
from IPython.display import display
import time, warnings

import pandas as pd

import time
get_ipython().run_line_magic('matplotlib', 'notebook')


# # Main code

# In[4]:


type_grocery = ['-','US Super', 'UK Super', 'US Hyper', 'AU Liquor', 'AU Drug', 'Convenience']
outlet = ['-','Store A', 'Store B', 'Store C', 'Store D', 'Store E', 'Store F', 'Store G', 'Store H', 'Store I', 'Store J', 'Store K' ,'Store L', 'None']

#mean_s ,sigma_s, mean_b, sigma_b, mean_a, sigma_a

parameter_dict = {'US Super': {'Store A': [2.92, 0.56, 0.95, 2.49, -0.8, 0.24],
                               'Store B': [2.78, 0.68, 1.45, 1.49, -0.41, 0.34],
                               'Store C': [2.91, 0.66, 1.30, 1.56, -0.55, 0.33],
                               'Store D': [2.73, 0.59, 1.41, 1.43, -0.28, 0.38],
                               'Store E': [3.13, 0.61, 1.62, 1.08, -0.19, 0.37],
                               'Store F': [2.44, 0.68, 1.39, 1.32, -0.33, 0.32],
                               'Store G': [2.72, 0.73, 0.54, 1.91, -0.39, 0.33],
                               'Store H': [2.89, 0.6, 1.26, 1.21, -0.5, 0.29],
                               'Store I': [2.31, 0.61, 1.18, 1.11, -0.59, 0.34],
                               'Store J': [2.85, 0.63, 0.8, 1.2, -0.48, 0.29],
                               'Store K': [2.92, 0.53, 1.12, 0.78, -0.31, 0.23]                               
                               },
#                   'AU Super': {'Store A': [2.63, 1, 3.57, 1.77, None, None],
#                                'Store B': [2.52, 0.94, 1.75, 1.61, None, None],
#                                'Store C': [2.92, 0.91, 1.82, 1.61, None, None],
#                                'Store D': [2.64, 0.97, 1.54, 2, None, None],
#                                'Store E': [3.09, 1.01, None, None, None, None]
#                               },
                  'UK Super': {'Store A': [3.17, 0.54, 2.32, 1.29, -0.95, 0.28]},
                  'US Hyper': {'Store A': [3.39, 0.68, 0.79, 2.68, -1, 0.3],
                               'Store B': [3.49, 0.6, 1.21, 1.46, -0.61, 0.23]                          
                              },
#                   'China Hyper':{'Store A': [3.75, 0.46, None, None, None, None, None]},
                  'AU Liquor': {'Store A': [1.51, 0.63, -1.32, 1.78, -0.89, 0.33],
                                'Store B': [1.41, 0.73, 0.52, 1.58, -0.9, 0.38],
                                'Store C': [1.52, 0.81, -0.58, 1.65, -1.04, 0.31],
                                'Store D': [1.36, 0.7, -1.32, 1.86, -1.11, 0.4],
                                'Store E': [1.15, 0.57, -0.76, 1.2, -1, 0.35],
                                'Store F': [1.15, 0.68, -1.37, 1.71, -1.03, 0.31],
                                'Store G': [1.13, 0.53, -1.27, 1.34, -1.01, 0.35],
                                'Store H': [1.19, 0.63, -1.08, 1.54, -1, 0.38],
                                'Store I': [1.16, 0.58, -1.63, 1.46, -0.96, 0.3],
                                'Store J': [0.91, 0.52, -1.53, 1.61, -1.17, 0.39],
                                'Store K': [0.97, 0.6, -1.77, 1.51, -1.04, 0.39],
                                'Store L': [1.25, 0.57, -1.68, 1.74, -0.94, 0.39]
                               },
                  'AU Drug': {'Store A': [1.71, 0.68, 0.45, 0.42, -0.51, 0.26],
                              'Store B': [1.73, 0.89, 0.3, 0.48, -0.57, 0.36],
                              'Store C': [1.64, 0.67, 0.22, 0.7, -0.6, 0.28],
                              'Store D': [1.46, 0.8, 0.1, 0.51, -0.56, 0.35],
                              'Store E': [1.56, 0.63, -0.11, 0.71, -0.55, 0.34],
                              'Store F': [1.49, 0.55, -0.85, 1.3, -0.61, 0.3],
                              'Store G': [1.22, 0.66, -0.85, 1.17, -0.71, 0.36]     
                              },
                'Convenience': {'None': [1.54, 0.7, 0.07, 0.76, -0.59, 0.32]}
#                   'Average small format': [1.34, 0.65, -0.76, 1.28, -0.85, 0.34],
#                   'Average': [2.15, 0.68, 0.31, 1.41, -0.71, 0.33]

                 }

layout_type = ["-","North", "East", "Pollock", "West", "Louie's"]


# In[5]:


def shelves_location(layout, num_rows,num_cols):
    layouts = []
    if layout == "North":
        for i in range(num_rows): #layout for right zone
            layouts.append((i,num_cols-1))
        layouts.remove((math.ceil(0.3*num_rows), num_cols-1))
        layouts.remove((math.ceil(0.9*num_rows), num_cols-1))

        for i in range(math.ceil(0.3*num_cols), num_cols): #layout for bottom zone
            layouts.append((num_rows-1, i))

        for i in range(math.ceil(0.25*num_rows)): #layout for left zone
            layouts.append((i,0))
    
        for j in range(math.ceil((1/3)*num_cols), math.ceil((2/3)*num_cols)+1): #middle layout
            layouts.append((0,j))
            layouts.append((2,j))
            layouts.append((3,j))
            layouts.append((5,j))
            layouts.append((6,j))
            layouts.append((9,j))
            layouts.append((10,j))
        #print(layouts)
        #print(0, math.ceil((1/3)*num_cols))
        #print(0, math.ceil((2/3)*num_cols))
        #print( math.ceil((1/3)*num_cols)+5)
        layouts.remove((0, math.ceil((1/3)*num_cols)+1))
        layouts.remove((0, math.ceil((1/3)*num_cols)+3))
        layouts.remove((0, math.ceil((1/3)*num_cols)+5)) 
        
        entrance = (math.ceil(0.30*num_rows)+1,0)
        cashier = (math.ceil(0.50*num_rows)+1,0)
        exit = (math.ceil(0.30*num_rows)+1,0)
        
        return layouts, entrance, cashier, exit
    
    elif layout == 'East':
        
        for i in range(num_rows): #layout for right zone
            layouts.append((i,num_cols-1))
        layouts.remove((math.ceil(0.5*num_rows), num_cols-1))
        layouts.remove((num_rows-1, num_cols-1))
        
        
        for i in range(math.ceil(0.8*num_rows)+1): #layout for left zone
            layouts.append((i,0))
        layouts.remove((math.ceil(0.4*num_rows),0))  
        
        for j in range(num_cols-1): #middle layout
            layouts.append((0,j))
        
        layouts.remove((0,math.ceil((1/3)*num_cols)+2))
        layouts.remove((0,math.ceil((1/3)*num_cols)+3))
        
        for i in range(1,num_rows-1):
            layouts.append((i,math.ceil((1/3)*num_cols)))
            layouts.append((i,math.ceil((1/3)*num_cols)+1))
            layouts.append((i,math.ceil((1/3)*num_cols)+4))
            layouts.append((i,math.ceil((1/3)*num_cols)+5))
            
        #added layout
        for i in range(math.ceil(0.7*num_rows),num_rows-1):
            layouts.append((i,math.ceil((1/3)*num_cols)+8))
            layouts.append((i,math.ceil((1/3)*num_cols)+9))
        
        for j in range(math.ceil(0.1*num_cols),math.ceil((1/3)*num_cols)-1):
            layouts.append((math.ceil(0.1*num_rows),j))
            layouts.append((math.ceil(0.1*num_rows)+1,j))
            layouts.append((math.ceil(0.1*num_rows)+3,j))
            layouts.append((math.ceil(0.1*num_rows)+4,j))
            
        for j in range(math.ceil(0.1*num_cols),math.ceil((1/3)*num_cols)-1):
            layouts.append((math.ceil(0.1*num_rows)+6,j))
            layouts.append((math.ceil(0.1*num_rows)+7,j))
            layouts.append((math.ceil(0.1*num_rows)+9,j))
            layouts.append((math.ceil(0.1*num_rows)+10,j))
        
        entrance = (num_rows-1,math.ceil(0.9*num_cols)-1)
        cashier = (num_rows-1,math.ceil(0.7*num_cols)-1)
        exit = (num_rows-1,math.ceil(0.5*num_cols)-1)
        
        return layouts, entrance, cashier, exit
        
    elif layout == 'Pollock':
        for j in range(math.ceil(0.4*num_cols), num_cols): #layout for bottom zone
            layouts.append((num_rows-1, j))
            
        for i in range(1, math.ceil(0.7*num_rows)+1): #layout for left zone
            layouts.append((i,0))
        
        for j in range(math.ceil(0.3*num_cols) ,math.ceil(0.7*num_cols)+1): #middle layout
            layouts.append((0,j))
            
        for j in range(math.ceil((1/3)*num_cols), math.ceil((2/3)*num_cols)+1): #middle layout
            for i in range(math.ceil(0.1*num_rows), math.ceil(0.8*num_rows)+1):
                layouts.append((i,j))
         
        for i in range(math.ceil(0.1*num_rows), math.ceil(0.8*num_rows)+1):
            layouts.remove((i, math.ceil((1/3)*num_cols) + 2))
            layouts.remove((i, math.ceil((2/3)*num_cols) - 2))
                
        layouts.remove((math.ceil(0.5*num_rows), math.ceil((1/3)*num_cols) + 3))
        layouts.remove((math.ceil(0.5*num_rows), math.ceil((1/3)*num_cols) + 4))
        layouts.remove((math.ceil(0.5*num_rows)+1, math.ceil((1/3)*num_cols) + 3))
        layouts.remove((math.ceil(0.5*num_rows)+1, math.ceil((1/3)*num_cols) + 4))
        
        layouts.remove((math.ceil(0.5*num_rows), math.ceil((2/3)*num_cols)))
        layouts.remove((math.ceil(0.5*num_rows), math.ceil((2/3)*num_cols)-1))
        layouts.remove((math.ceil(0.5*num_rows)+1, math.ceil((2/3)*num_cols)))
        layouts.remove((math.ceil(0.5*num_rows)+1, math.ceil((2/3)*num_cols)-1))
        
        for j in range(math.ceil((0.8)*num_cols),math.ceil((0.8)*num_cols)+2): #middle layout
            for i in range(math.ceil(0.3*num_rows), math.ceil(0.7*num_rows)+1):
                layouts.append((i,j))
                
        for j in range(math.ceil((0.8)*num_cols),math.ceil((0.8)*num_cols)+2):       
            layouts.append((num_rows-2,j))
        
        entrance = (num_rows-1, math.ceil(0.25*num_cols) -1)
        cashier = (num_rows-1, math.ceil(0.1*num_cols) -1)
        exit = (num_rows-1, math.ceil(0.25*num_cols) -1)
        
        return layouts, entrance, cashier, exit
    
    elif layout == "West":
        for i in range(math.ceil(0.5*num_rows)): #layout for right zone
            layouts.append((i,num_cols-1))
        layouts.append((math.ceil(0.5*num_rows) + 1,num_cols-1))
        
        for i in range(math.ceil(0.2*num_cols), math.ceil(0.7*num_cols)): #layout for bottom zone
            layouts.append((num_rows-1, i))
            
        for i in range(math.ceil(0.05*num_rows), math.ceil(0.95*num_rows)): #layout for left zone
            layouts.append((i,0))
            
        for j in range(math.ceil((1/3)*num_cols), math.ceil((2/3)*num_cols)+1): #middle layout
            layouts.append((0,j))
        
        for i in range(2,math.ceil(0.7*num_rows)):
            layouts.append((i,math.ceil((1/3)*num_cols)))
            layouts.append((i,math.ceil((1/3)*num_cols)+1))
            layouts.append((i,math.ceil((1/3)*num_cols)+4))
            layouts.append((i,math.ceil((1/3)*num_cols)+5))
       
        for j in range(math.ceil((0.2)*num_cols), math.ceil((2/3)*num_cols)+2): #middle layout
            layouts.append((math.ceil(0.7*num_rows) + 1,j))
            layouts.append((math.ceil(0.7*num_rows) + 2,j))
            
        layouts.remove((math.ceil(0.7*num_rows) + 1, math.ceil((1/3)*num_cols)+2))
        layouts.remove((math.ceil(0.7*num_rows) + 1, math.ceil((1/3)*num_cols)+3))
        layouts.remove((math.ceil(0.7*num_rows) + 2, math.ceil((1/3)*num_cols)+2))
        layouts.remove((math.ceil(0.7*num_rows) + 2, math.ceil((1/3)*num_cols)+3))
        
        entrance = (num_rows-1, math.ceil(0.7*num_cols)+1)
        cashier = (num_rows-1, math.ceil(0.8*num_cols)+1)
        exit = (num_rows-1, math.ceil(0.7*num_cols)+1)
        
        return layouts, entrance, cashier, exit
    
    elif layout == "Louie's":
        
        for i in range(num_rows): #layout for right zone
            layouts.append((i,num_cols-1))
        
        for i in range(num_rows): #layout for left zone
            layouts.append((i,0))
        layouts.remove((math.ceil(0.5*num_rows),0))
        
        for j in range(1,num_cols-1): #layout for upper zone
            layouts.append((0, j))
            
        for j in range(math.ceil((0.10)*num_cols), math.ceil((0.20)*num_cols)+1): #middle layout
            for i in range(math.ceil(0.2*num_rows), math.ceil(0.5*num_rows)+1):
                layouts.append((i,j))
                
        for j in range(math.ceil((0.30)*num_cols), math.ceil((0.40)*num_cols)+1): #middle layout
            for i in range(math.ceil(0.1*num_rows), math.ceil(0.7*num_rows)+1):
                layouts.append((i,j))     
        
        for j in range(math.ceil((0.55)*num_cols), math.ceil((0.65)*num_cols)+1): #middle layout
            for i in range(math.ceil(0.1*num_rows), math.ceil(0.7*num_rows)+1):
                layouts.append((i,j)) 
        
        for j in range(math.ceil((0.75)*num_cols), math.ceil((0.85)*num_cols)+1): #middle layout
            for i in range(math.ceil(0.2*num_rows), math.ceil(0.5*num_rows)+1):
                layouts.append((i,j)) 
            
        for j in range(math.ceil((0.30)*num_cols), math.ceil((0.65)*num_cols)+1): #middle layout
            layouts.append((math.ceil(0.7*num_rows)+2, j)) 
        
        entrance = (num_rows-1,math.ceil(0.9*num_cols)-1)
        cashier = (math.ceil(0.7*num_rows)+3, math.ceil((0.30)*num_cols))
        exit = (num_rows-1,math.ceil(0.2*num_cols)-1)
        
        return layouts, entrance, cashier, exit
        
        
            


# In[6]:


# num_rows = 20
# num_cols = 20
# layouts = shelves_location(layout ="Louie's", num_rows = num_rows, num_cols = num_cols)[0]
# # #print(layouts)
# grid = np.zeros((num_rows,num_cols), dtype=int)
# for pos in layouts:
#     #print(pos)
#     x, y = pos
#     grid[x][y] = 1  # or any other value you want to set


# In[7]:


#print(grid)


# In[8]:


# size = (20, 20)

# # Generate a 2D array with random numbers using random.uniform
# random_array = np.random.uniform(low=0.0, high=1.0, size=size)


# In[9]:


# random_array[0][0]


# In[10]:


# random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])


# In[11]:


# # import numpy as np

# def manhatthan_distance(point1, point2):
#     """Calculate the Euclidean distance between two points."""
#      return (point2[0] - point1[0], point2[1] - point1[1])

# def min_distance(points, init_point):
# #     """Find the minimum distance between points."""
#     min_dist = float('inf')
#     min_points = None
#     for i in range(len(points)):
#         dist = np.abs(manhatthan_distance(init_point, points[i]))[0] + np.abs(manhatthan_distance(init_point, points[i]))[1]
#         #print(dist)
#         if dist < min_dist:
#             min_dist = dist
#             min_dist_point = manhatthan_distance(init_point, points[i])
#             min_points = (points[i][0], points[i][1])
#     return min_dist_point, min_points

# def discretize_point(point):
#     """Discretize a point into multiple points with each coordinate being either (-1, 0) or (0, 1)."""
#     discretized_points = []
#     for i in range(abs(point[0])):
#         if point[0] < 0:
#             discretized_points.append((-1, 0))
#         else:
#             discretized_points.append((1, 0))
#     for i in range(abs(point[1])):
#         if point[1] < 0:
#             discretized_points.append((0, -1))
#         else:
#             discretized_points.append((0, 1))
#     return discretized_points


# In[12]:


# def find_total_steps(input_list, row, col):
#     total_steps= []
#     while len(input_list) != 0:
#         min_dist, min_points = min_distance(input_list,(row,col))
#         dis_point = discretize_point(min_dist)
#         total_steps.extend(dis_point)
#         for (row_step, col_step) in da:
#             row, col = row + row_step, col + col_step
        
#         input_list.remove((row,col))
#     return len(total_steps)


# In[13]:


# points = [(5, 9), (18, 5), (17, 5), (14, 8), (9, 6), (4, 8), (13, 14), (8, 1), (3, 13), (7, 6), (7, 2), (2, 5), (14, 7), (1, 0), (1, 12), (9, 12), (2, 9), (12, 0), (19, 3), (19, 19), (4, 18), (6, 2), (11, 7), (9, 17), (13, 0), (9, 11), (3, 10), (18, 6), (2, 2), (17, 1), (1, 1), (17, 3), (17, 10), (6, 18), (0, 0), (11, 15), (15, 5), (10, 18), (8, 6), (10, 11), (15, 16), (12, 14), (11, 3), (16, 14), (11, 2), (16, 15), (10, 19), (12, 2), (19, 6), (14, 19), (14, 16), (18, 18), (7, 8), (5, 7), (10, 8), (12, 4), (14, 12), (1, 4), (11, 9), (10, 2), (9, 5), (13, 19), (15, 18), (5, 17), (3, 7), (1, 5), (13, 7), (6, 12), (17, 15), (18, 9), (8, 12), (3, 12), (16, 13), (16, 16), (10, 10), (8, 16), (8, 3), (12, 13), (4, 3), (15, 14), (0, 16), (16, 1), (0, 2), (3, 8), (19, 8), (9, 10), (11, 17), (2, 11), (0, 5), (7, 17), (1, 8), (0, 10), (4, 1), (2, 7), (11, 18), (6, 13), (0, 3), (16, 2), (5, 6), (9, 7), (2, 17), (5, 19), (3, 18), (3, 11), (14, 9), (19, 15), (0, 18), (18, 0), (8, 9), (13, 11), (19, 4), (6, 10), (10, 15), (0, 13)]

# find_discretize_length(points,row= 5,col=0)


# In[14]:


# points = [(5, 9), (18, 5), (17, 5), (14, 8), (9, 6), (4, 8), (13, 14), (8, 1), (3, 13), (7, 6), (7, 2), (2, 5), (14, 7), (1, 0), (1, 12), (9, 12), (2, 9), (12, 0), (19, 3), (19, 19), (4, 18), (6, 2), (11, 7), (9, 17), (13, 0), (9, 11), (3, 10), (18, 6), (2, 2), (17, 1), (1, 1), (17, 3), (17, 10), (6, 18), (0, 0), (11, 15), (15, 5), (10, 18), (8, 6), (10, 11), (15, 16), (12, 14), (11, 3), (16, 14), (11, 2), (16, 15), (10, 19), (12, 2), (19, 6), (14, 19), (14, 16), (18, 18), (7, 8), (5, 7), (10, 8), (12, 4), (14, 12), (1, 4), (11, 9), (10, 2), (9, 5), (13, 19), (15, 18), (5, 17), (3, 7), (1, 5), (13, 7), (6, 12), (17, 15), (18, 9), (8, 12), (3, 12), (16, 13), (16, 16), (10, 10), (8, 16), (8, 3), (12, 13), (4, 3), (15, 14), (0, 16), (16, 1), (0, 2), (3, 8), (19, 8), (9, 10), (11, 17), (2, 11), (0, 5), (7, 17), (1, 8), (0, 10), (4, 1), (2, 7), (11, 18), (6, 13), (0, 3), (16, 2), (5, 6), (9, 7), (2, 17), (5, 19), (3, 18), (3, 11), (14, 9), (19, 15), (0, 18), (18, 0), (8, 9), (13, 11), (19, 4), (6, 10), (10, 15), (0, 13)]
# # min_dist, min_points = min_distance(points,(4,13))
# # print(min_dist)
# a = []
# for item in points:
#     a.extend(discretize_point(item))

# print(a)


# In[78]:


#Function to generate the position of row(sub_category) that customer will go to buy per column(shelf)
def selected_row_for_each_col(grid_row, num_items_to_buy):
    if num_items_to_buy <= grid_row:
        return sorted(random.sample([i for i in range(0,grid_row)],num_items_to_buy))
    else :
        #can be repeated
        return sorted(random.choices([i for i in range(0,grid_row)],k=num_items_to_buy))
    
#Function to generate correlated parameters x_s, x_b, x_a for each customer
def correlated_samples(r_matrix,mean_s,sigma_s,mean_b,sigma_b,mean_a,sigma_a,size):
    poisson_para = dist.rand(dist.LogNormal(mean_b,sigma_b))
    margins = [dist.LogNormal(mean_s,sigma_s), #shopping_time
               dist.Poisson(poisson_para), #basket size
               dist.LogitNormal(mean_a,sigma_a)]  #proportion of area visit
    #adjusted_corr = bs.pearson_match(r_matrix, margins)
    #print("adjusted_corr=", adjusted_corr)
    x = bs.rvec(size, r_matrix, margins)
    #print(x)
    #x = x.flatten()
    #print(x)
    #print(x[0])
    return x

#Find distance between 2 points in terms of x and y
def manhatthan_distance(point1, point2):
    return (point2[0] - point1[0], point2[1] - point1[1])

#Check each point in set from initial point and find minimum distance to find number of step
def min_distance(points, init_point):
    min_dist = float('inf')
    min_points = None
    for i in range(len(points)):
        dist = np.abs(manhatthan_distance(init_point, points[i]))[0] + np.abs(manhatthan_distance(init_point, points[i]))[1]
        #print(dist)
        if dist < min_dist:
            min_dist = dist
            min_dist_point = manhatthan_distance(init_point, points[i])
            min_points = (points[i][0], points[i][1])
    return min_dist_point, min_points

#Algorithm to make steps
def discretize_point(point):
    discretized_points = []
    for i in range(abs(point[0])):
        if point[0] < 0:
            discretized_points.append((-1, 0))
        else:
            discretized_points.append((1, 0))
    for i in range(abs(point[1])):
        if point[1] < 0:
            discretized_points.append((0, -1))
        else:
            discretized_points.append((0, 1))
    return discretized_points


def find_total_steps(input_list, row, col, pos_list, cashier_pos, exit_pos, x_b):
    steps= []
    while len(input_list) != 0:
        min_dist, min_points = min_distance(input_list,(row,col))
        dis_point = discretize_point(min_dist)
        steps.extend(dis_point)
        for (row_step, col_step) in dis_point:
            row, col = row + row_step, col + col_step
        
        input_list.remove((row,col))
        if len(input_list) == 1:
            pos_list = input_list[0]
    #print("steps before finish buying",len(steps))        
    to_cashier_steps = np.abs(manhatthan_distance(pos_list, cashier_pos)[0]) + np.abs(manhatthan_distance(pos_list, cashier_pos)[1])
    to_exit_steps = np.abs(manhatthan_distance(pos_list, exit_pos)[0]) + np.abs(manhatthan_distance(pos_list, exit_pos)[1])
    if x_b == 0:
        total_steps = len(steps) + to_exit_steps
    else:
        total_steps = len(steps) + to_cashier_steps
    #print("steps after finish buying",total_steps)  
        
    return total_steps


#Function for customers activity
def customer(env,name,shelves,cashiers,num_on_shelf,count,cust_dict,warmup_time,shelves_layout,layout_widget,SIM_TIME,monitor_time):
        x_s = cust_dict[name]['x_s']
        x_b = cust_dict[name]['x_b']
        x_a = cust_dict[name]['x_a']
        #print(x_s,x_b,x_a)
        # store layout
        grid_row = len(cust_dict['customer 1']['grid'])
        grid_col = len(cust_dict['customer 1']['grid'][0])


        #print(f"{name} has basketsize = {x_b}")
        #print(f'{name}: Arrives at time {env.now:.2f} has total time of {x_s}')
        
        #cust_dict[name]['time'].append(env.now) #Arrival timestamp

        # number of segments to cover (need at least 1 segment)
        v = max(1,round(x_a*(grid_col*grid_row)))
        #print("x_b =", x_b)
        #print("v =", v)
        #print("x_s =", x_s)
        
        # number of buying item per segment use multinomial random with equally probability = 1/v
        cust_dict[name]['buy_item_per_seg'] = np.random.multinomial(x_b, [1/v]*v).flatten().tolist()
        #print(f"{name} needs to buy {cust_dict[name]['buy_item_per_seg']}")
        
        # prepare number of items to buy per segment in iterary
        #buy_item_iter = iter(cust_dict[name]['buy_item_per_seg'])
        
        # count number of aisles that customer just roaming around without buying anything (use to specify chill time)
        cust_dict[name]['count_0'] = cust_dict[name]['buy_item_per_seg'].count(0)
        
        if x_b == 0: #customer can visit shelf without buying anythin
            cust_dict[name]['selected_column'] = random.sample([(i,j) for j in range(0,grid_col) for i in range(0,grid_row)], v)
            
            
        elif x_b > 0: #customer cannot visit shelf without buying anything
            #Need to check minimum number of shelves needed for each customer to visit
            items = [item for item in cust_dict[name]['buy_item_per_seg'] if item != 0]
            #print("len_items =", len(items))
            no_shelf_positions = [(i, j) for j in range(grid_col) for i in range(grid_row) if (i, j) not in shelves_layout]
                    
            if v > len(shelves_layout):
                if len(items) > len(shelves_layout):
                    #print("items > shelve layouts")
                    available_positions = random.choices(shelves_layout, k = len(items)) #shelves can be repeated
                    extra_positions = random.sample(no_shelf_positions, v - len(items))
                    available_positions.extend(extra_positions)
                    random.shuffle(available_positions)
                    cust_dict[name]['selected_column'] = available_positions
                    #print(name, "need to visit area", cust_dict[name]['selected_column'])
                else:
                    #print("items < shelve layout")
                    available_positions = random.sample(shelves_layout, len(items)) #shelves should not be repeated
                    extra_positions = random.sample(no_shelf_positions, v - len(items))
                    available_positions.extend(extra_positions)
                    random.shuffle(available_positions)
                    cust_dict[name]['selected_column'] = available_positions
                    #print(name, "need to visit area", cust_dict[name]['selected_column'])
                                    
            else:
                #print("v < shelve layout")
               
                shelf_need = random.sample(shelves_layout, len(items))
               # print("len_shelf_need =", len(shelf_need))
                no_shelf_positions = [(i, j) for j in range(grid_col) for i in range(grid_row) if (i, j) not in shelves_layout]
                others_need = random.sample(no_shelf_positions, v - len(items))
                #print("len other need =",len(others_need))
                available_positions = []
                available_positions.extend(shelf_need)
                available_positions.extend(others_need)
                random.shuffle(available_positions)
                #print(available_positions)
                #print(len(available_positions))
                cust_dict[name]['selected_column'] = available_positions
                
        #print(name, "need to visit area", cust_dict[name]['selected_column'])
        
        for i,j in cust_dict[name]['selected_column']:
            cust_dict[name]['grid'][i][j] = True #locations that customers must visit
    
        #Algorithm for timing customer on every step
        time_per_step = 0.625/60
        def find_current_pos(env, cust_grid, row, col, cust_segment):
            row_step, col_step = (0,0)
            #print("Number of grid that is True =",len([item for row in grid for item in row if item is True]))
            #print("Those gird are", [(row, col) for row in range(len(grid)) for col in range(len(grid[0])) if grid[row][col] is True")
            #print("num cust segment =", len(cust_segment))
            #print(cust_segment)
            pos_show_more_than_one = [entity for entity, count in Counter(cust_segment).items() if count > 1]
            #print("pos show more than one is =", pos_show_more_than_one )
#             walking_speed = 500 #m/min
#             step_distance = 0.7 #m
#             time_per_step = step_distance/walking_speed
            min_dist, min_points = min_distance(cust_segment,(row,col))
            #print("enter row, enter col =", (row, col))
            #print(min_dist, min_points)
            discrete_point = discretize_point(min_dist)

            for (row_step, col_step) in discrete_point:
                row, col = row + row_step, col + col_step
                yield env.timeout(time_per_step)
                             
                #print(f"{name} at {row, col} at time {env.now:.2f}")
                cust_dict[name]['time'].append((round(env.now),(row,col)))
                #print("row_inside =",row, "col_inside =",col)
                #cust_dict[name]['total_steps']+=1
        
            if (row,col) in pos_show_more_than_one:
                cust_grid[row][col] = True
            else:
                cust_grid[row][col] = False
        
            #print("row , col=", (row,col))
            #print("After break while loop, Grid", (row,col), "must be", grid[row][col])
            #print(row,col)
            yield row,col
            
        #Random walk algorithm
        ini_row, ini_col = shelves_location(layout=layout_widget.value, num_rows=grid_row, num_cols=grid_col)[1]
        #ini_pos_vincinity = [(ini_row, ini_col), (ini_row+1, ini_col), (ini_row-1, ini_col), (ini_row, ini_col+1), (ini_row, ini_col-1), (ini_row+1, ini_col-1), (ini_row-1, ini_col-1), (ini_row+1, ini_col+1), (ini_row-1, ini_col+1)]
        row, col = ini_row,ini_col
        
        #arrival time and position
        cust_dict[name]['time'].append((round(env.now),(ini_row,ini_col))) #Arrival timestamp
        cust_dict[name]['arrive_time'] = env.now
        #print(f'{name}: Arrives at time {env.now:.2f} has total time of {x_s}')
        
        cust_seg_copy = cust_dict[name]['selected_column'].copy()
        #Find number of total steps before hand so that I can use it to deduct from shopping time
        
        #cashier position
        cashier_pos = shelves_location(layout=layout_widget.value, num_rows=grid_row, num_cols=grid_col)[2]
        
        #exit from store position
        exit_pos = shelves_location(layout=layout_widget.value, num_rows=grid_row, num_cols=grid_col)[3]
        
        cust_dict[name]['total_steps'] = find_total_steps(input_list=cust_seg_copy, row=row, col=col, pos_list=cust_dict[name]['last_position'], cashier_pos=cashier_pos, exit_pos=exit_pos, x_b=x_b)
        #print(f"{name} total steps = {cust_dict[name]['total_steps']}")
        
        #Algorithm to assign customer to area
        #new_row, new_col = 0,0
        #print(row,col)
        while cust_dict[name]['num_seg_covered'] <= v: #continue shopping until v segments are covered           
            while len(cust_dict[name]['selected_column']) != 0:
                #print("num items length =", len(cust_dict[name]['buy_item_per_seg']))
                #print("new row, new col =", (new_row, new_col))
                shelf_list = [item for item in cust_dict[name]['selected_column'] if item in shelves_layout]
                #print(f"shelf length for {name} =", len(shelf_list))
                not_shelf_list = [item for item in cust_dict[name]['selected_column'] if item not in shelves_layout] 
                #print(f"not shelf length for {name} =", len(not_shelf_list))
                
                result = find_current_pos(env=env, 
                                          cust_grid=cust_dict[name]['grid'], 
                                          row=row, 
                                          col=col,  
                                          cust_segment = cust_dict[name]['selected_column'])
                     
                for item in result:
                    #print(item)
                    if isinstance(item, tuple):
                        pos = item
                        #yield grid[item[0]][item[1]].request()
                    elif isinstance(item, simpy.Timeout):
                        time_out_object = item
                        yield time_out_object
                                      
                #print(f"{name} current position at {pos} at time {env.now:.2f} with total steps = {cust_dict[name]['total_steps']}")
                #yield grid[pos[0]][pos[1]].request()
                #randomly choose column with equal probaility from the selected column
                #col = random.choice(cust_dict[name]['selected_column'])
                
                #Determine shopping time for each segment (depend on number of items to buy in each segment)
                #If basket size > 0 (custs either buy things or not)
                
                #Time indication part
                if x_b > 0:
                    if pos in not_shelf_list:
                        # Find any entity that is zero in list A
                        num_items_to_buy = next((entity for entity in cust_dict[name]['buy_item_per_seg'] if entity == 0), None)

                    elif pos in shelf_list:
                        # Find any entity that is not zero in list A
                        num_items_to_buy = next((entity for entity in cust_dict[name]['buy_item_per_seg'] if entity != 0), None)

                    #CASE 1: time spent for custs buy at least 1 item in any aisle
                    #print(f"{name} num item to buy = {num_items_to_buy}")
                    if all(item > 0 for item in cust_dict[name]['buy_item_per_seg']):
                        #buy_time_per_seg = round((num_items_to_buy / x_b) * x_s, 2)
                        cust_dict[name]['buy_time_per_seg'] = (num_items_to_buy / x_b) * (x_s-(cust_dict[name]['total_steps']*time_per_step))
                    #CASE 2: time spent for custs might not buy or buy anything in some aisles. Need chill time    
                    else:
                        chill_time = 0.15 * (x_s - (cust_dict[name]['total_steps']*time_per_step))  # Total time that customer spends without buying anything

                        #CASE 2.1: time spent at aisles that custs buy something 
                        if num_items_to_buy > 0:
                            #buy_time_per_seg = round((num_items_to_buy/x_b)*(x_s-chill_time),2)
                            cust_dict[name]['buy_time_per_seg'] = (num_items_to_buy/x_b)*(x_s - (cust_dict[name]['total_steps']*time_per_step) - chill_time)
                            #print(f"{name} supposed to be buy time per seg when x_b > 0", buy_time_per_seg)
                           
                        #CASE 2.2: time spent at aisles that custs don't buy anything    
                        else:
                            #buy_time_per_seg = round(chill_time/cust_dict[name]['count_0'],2)
                            
                            cust_dict[name]['buy_time_per_seg'] = chill_time/cust_dict[name]['count_0']
                            #print(f"{name} supposed to be buy time per seg when x_b > 0 but item to buy is 0 =", buy_time_per_seg)
                           
                    #print("buy time =", buy_time_per_seg)
                else:
                    #buy_time_per_seg = round(x_s / v, 2)
                    cust_dict[name]['buy_time_per_seg']  = (x_s-(cust_dict[name]['total_steps']*time_per_step))/v
                    #print(f"{name} supposed to be buy time per seg when x_b is 0 =", buy_time_per_seg)
                
                #print(f"{name} supposed to be buy time per seg ", cust_dict[name]['buy_time_per_seg'])
                
                if cust_dict[name]['buy_time_per_seg']  <= 0:
                    cust_dict[name]['neg_buy_time'] = True
                    break
                
                else:
            
                    if x_b > 0:
                        
                        if num_items_to_buy == 0:
                            #print(f"x_b > 0 and num item = 0. Cus {name} go to not shelf coordinate")
                                    #pos = random.choice(not_shelf)
                                    #print(f"{name} is at {pos} at time {env.now:.2f} spend time {buy_time_per_seg} min without buying anything")
                            cust_dict[name]['time'].append((round(env.now),pos))
                            #print("yield buy time per seg when x_b not 0 but num item is 0 =", cust_dict[name]['buy_time_per_seg'])
                            yield env.timeout(cust_dict[name]['buy_time_per_seg'])
                            cust_dict[name]['time'].append((round(env.now),pos))
            #                         if env.now >= warmup_time:
            #                             customers_queue_for_shelf = [[0 for j in range(grid_col)]for i in range(grid_row)]

                        else:
                                #pos = random.choice(shelf)
                                #print("x_b > 0 & num_items_to_buy > 0")

                            shelf = shelves[pos[0]][pos[1]]

                            #print(f'{name}: arrive at shelf {pos} at time {env.now:.2f}')
                            cust_dict[name]['time'].append((round(env.now),pos))
                            arrive_shelf_time = env.now
                                #collect position at time customer arrive at position
                                #grid[pos[0]][pos[1]]+=1

                            with shelf["resource"].request() as shelf_req:
                                yield shelf_req
                                
                                num_items_for_each_shelf = num_items_to_buy
                                yield shelf["container"].get(num_items_for_each_shelf)
                                
                                #print(f'{name}: get shelf {pos} at time {env.now:.2f}')
                                
                                get_shelf_time = env.now
                                wait_shelf_time = get_shelf_time - arrive_shelf_time
                                
                                cust_dict[name]['waiting_time_for_shelf'] += wait_shelf_time
                                
                                #print(f'{name}: wait for shelf {pos} for {wait_shelf_time} min')
                                #print(f'{name}: wait for shelf for {cust_dict[name]["waiting_time_for_shelf"]} min')
                                    #print(f'{name}: get shelf {pos} at time {env.now:.2f}')
                                cust_dict[name]['time'].append((round(env.now),pos))
                                    #print(buy_time_per_seg/num_items_to_buy)
                                yield env.timeout(cust_dict[name]['buy_time_per_seg'])
                                    #print(f'{name}: finish buying shelf {pos} at time {env.now:.2f}')

                                cust_dict[name]['time'].append((round(env.now),pos))


                                #This part call the refill process from refill function
                            refill_level = 45 #threshold to refill
                            if shelf['container'].level <= refill_level:

                                yield env.process(shelf_refill_process(env=env,name=name,shelf=shelf,num_on_shelf=num_on_shelf))# wait for fridge refilling process to be ended

                                count['refill_count'][pos[0]][pos[1]] +=1

                        cust_dict[name]['buy_item_per_seg'].remove(num_items_to_buy)

                    else:

                        if pos in shelves_layout:
                            #print("x_b = 0 & shelf")
                            cust_dict[name]['time'].append((round(env.now),pos))
                            #print(f'{name}: arrive at shelf at time {env.now:.2f}')
                            arrive_shelf_time = env.now
                            with shelves[pos[0]][pos[1]]['resource'].request() as shelves_req:
                                yield shelves_req
                                #print(f'{name}: get shelf {pos} at time {env.now:.2f}')
                                cust_dict[name]['time'].append((round(env.now),pos))
                                
                                get_shelf_time = env.now
                                wait_shelf_time = get_shelf_time - arrive_shelf_time
                                cust_dict[name]['waiting_time_for_shelf'] += wait_shelf_time
                                
                                #print(f'{name}: wait for shelf {pos} for {wait_shelf_time} min')
                                #print(f'{name}: wait for shelf for {cust_dict[name]["waiting_time_for_shelf"]} min')
                                
                                yield shelves[pos[0]][pos[1]]["container"].get(1)
                                yield shelves[pos[0]][pos[1]]["container"].put(1)#customer won't buy anything
                                #print("yield buy time per seg when x_b is 0 =", cust_dict[name]['buy_time_per_seg'])
                                yield env.timeout(cust_dict[name]['buy_time_per_seg'])


                            cust_dict[name]['time'].append((round(env.now),pos))
                        else:
                            #print("x_b = 0 & not shelf")
                            #print(f"{name}: visit not shelf {pos} at time {env.now:.2f}")
                            cust_dict[name]['time'].append((round(env.now),pos))
                            yield env.timeout(cust_dict[name]['buy_time_per_seg'])                        

                            cust_dict[name]['time'].append((round(env.now),pos))
                            
                global customers_queue_for_shelf
                global customers_users_on_shelf
                
                if env.now >= warmup_time:
                    customers_queue_for_shelf = [[len(shelves[i][j]["resource"].queue) if shelves[i][j] != 0 else 0 
                                                        for j in range(grid_col)]for i in range(grid_row)]
                    customers_users_on_shelf = [[len(shelves[i][j]["resource"].users) if shelves[i][j] != 0 else 0 
                                                        for j in range(grid_col)]for i in range(grid_row)]
                

                row, col = pos
                    #print("row_2, col_2 =", row,col)

                    #print("It is time to remove pos", pos)
                cust_dict[name]['selected_column'].remove(pos)
                    #print(f"There are {len(cust_dict[name]['selected_column'])} segments left for {name}")
                    #print(f"That pos is {cust_dict[name]['selected_column']} for {name}")
                    #record last position
                if len(cust_dict[name]['selected_column']) == 1:
                        cust_dict[name]['last_position'] = cust_dict[name]['selected_column'][0]
                    
            
            if cust_dict[name]['neg_buy_time']:
                #print(f'{name} has neg buy time')
                #cust_dict.pop(name)
                break
                
                
            cust_dict[name]['num_seg_covered'] +=1
       
        #print("cashier pos = ", cashier_pos)
        #print("exit pos = ", exit_pos)
        #print(cust_dict[name]['neg_buy_time'])
        if cust_dict[name]['neg_buy_time']:
            #print(f'{name} has neg buy time')
            cust_dict[name]['shopping_time'] = np.NaN
            #continue
            
        else:
            
            if x_b == 0:
                #print("exit pos = ", exit_pos)
                #print(f"{name} last position is {cust_dict[name]['last_position']}")
                exit_dist = manhatthan_distance(cust_dict[name]['last_position'], exit_pos)
                exit_move = discretize_point(exit_dist)
                for r, c in exit_move:
                    x, y = cust_dict[name]['last_position']
                    cust_dict[name]['last_position'] = (x + r, y + c)
                    yield env.timeout(time_per_step)
                    cust_dict[name]['time'].append((round(env.now),cust_dict[name]['last_position']))
                    #print(f"{name} move to {cust_dict[name]['last_position']} to leave store at time {env.now:.2f}")

                leave_time = env.now
                cust_dict[name]['shopping_time'] = leave_time - cust_dict[name]['arrive_time']
                cust_dict[name]['departure_time'] = env.now
                #print(f"{name} leave store at time {env.now:.2f} at position {cust_dict[name]['last_position']}")
            else:
                cashier_dist = manhatthan_distance(cust_dict[name]['last_position'], cashier_pos)
                cashier_move = discretize_point(cashier_dist)
                for r, c in cashier_move:
                    x, y = cust_dict[name]['last_position']
                    cust_dict[name]['last_position'] = (x + r, y + c)
                    yield env.timeout(time_per_step)

                    cust_dict[name]['time'].append((round(env.now),cust_dict[name]['last_position']))
                    #print(f"{name} move to {cust_dict[name]['last_position']} to cashier at time {env.now:.2f}")
                catch_time = env.now
                cust_dict[name]['shopping_time'] = catch_time - cust_dict[name]['arrive_time']

                # Time spent walking to cashier (assume constant for now)
                #walking_time_to_cashier = 1 
                #yield env.timeout(walking_time_to_cashier)
                #print(f'{name}: arrive at cashier at time {env.now:.2f}')
                #print(f"{name} arrive at cashier at time {env.now:.2f} at position {cust_dict[name]['last_position']}")
                cust_dict[name]['arrive_cashier_time'] = env.now
        #         print(name)
        #         print(cust_dict[name]['arrive_cashier_time'])
                #global cust_arrival_at_cashier
                global customers_queue_for_cashier 

                if env.now >= warmup_time:
                    #cust_arrival_at_cashier.append(cust_dict[name]['arrive_cashier_time'])
                    customers_queue_for_cashier.append((round(env.now - warmup_time,2), len(cashiers.queue)))

                with cashiers.request() as cashier_req:
                    yield cashier_req
                    #print(f'{name}: start taking service from cashier {cashiers.queue} at time {env.now:.2f}')
                    cust_dict[name]['get_cashier_time'] = env.now
                    cust_dict[name]['time'].append((round(env.now),cust_dict[name]['last_position']))

                    cust_dict[name]['waiting_time_for_cashier'] = cust_dict[name]['get_cashier_time'] - cust_dict[name]['arrive_cashier_time']
                    #print(f"{name} buy {x_b}")
                    payment_time = np.random.exponential(scale=0.05*x_b, size=1)[0]
                    #print(f"{name} spent {payment_time} on cashier")
                    yield env.timeout(payment_time)
                    cust_dict[name]['time'].append((round(env.now),cust_dict[name]['last_position']))


                #print(f"{name} finished payment  at cashier at time {env.now:.2f} at position {cust_dict[name]['last_position']}")

                exit_dist_from_cashier = manhatthan_distance(cust_dict[name]['last_position'], exit_pos)
                exit_move_from_cashier = discretize_point(exit_dist_from_cashier)
                for r, c in exit_move_from_cashier:
                    x, y = cust_dict[name]['last_position']
                    cust_dict[name]['last_position'] = (x + r, y + c)
                    yield env.timeout(time_per_step)
                    cust_dict[name]['time'].append((round(env.now),cust_dict[name]['last_position']))

                cust_dict[name]['departure_time'] = env.now
                #print(f"{name} leave store at {env.now:.2f} at position {cust_dict[name]['last_position']}")
            

# Function to generate customers
def customer_generator(env,lambd,x,shelves,cashiers,num_cols,num_rows,num_on_shelf,count,warmup_time,shelves_layout,layout_widget,SIM_TIME,monitor_time):
    cust_number = 1
    global cust_dict
    while True:
        inter_arrival_time = np.random.exponential(scale=1/lambd, size=1)[0]
    #while cust_number < 5:
        if inter_arrival_time >=0:
            yield env.timeout(inter_arrival_time)
            customer_name = f'customer {cust_number}'
            
            cust_dict[customer_name]={'grid':[[False for _ in range(num_cols)] for _ in range(num_rows)],
                                     'num_seg_covered': 1,
                                     'selected_column': 0,
                                     'selected_row': 0,
                                     'buy_item_per_seg': 0,
                                     'buy_time_per_seg': 0,
                                     'count_0': 0,
                                     'time': [],
                                     'arrive_cashier_time':0,
                                     'get_cashier_time':0,
                                     'waiting_time_for_cashier':0,
                                     'x_s':x[cust_number-1][0],
                                     'x_b':x[cust_number-1][1],
                                     'x_a':x[cust_number-1][2],
                                     'total_steps': 0,
                                     'last_position': 0,
                                     'arrive_time': 0,
                                     'departure_time': 0,
                                     'waiting_time_for_shelf':0,
                                     'shopping_time': 0,
                                     'basket_size':x[cust_number-1][1],
                                     'neg_buy_time': False
                                     }
        else:
            print("Wrong input. Try using positive arrival rate")
            break
        
#         print(customer_name, cust_dict[customer_name]['x_s'])
#         print(customer_name, cust_dict[customer_name]['x_b'])
#         print(customer_name, cust_dict[customer_name]['x_a'])
        #print(f'{customer_name}: Arrives at time {env.now:.2f}')
        env.process(customer(env=env,
                             name=customer_name,
                             shelves=shelves,
                             cashiers=cashiers,
                             num_on_shelf=num_on_shelf,
                             count=count,
                             cust_dict=cust_dict,
                             warmup_time=warmup_time,
                             shelves_layout=shelves_layout,
                             layout_widget=layout_widget,
                             SIM_TIME=SIM_TIME,
                             monitor_time=monitor_time))
        #print(cust_number)
        cust_number += 1

            
#Function to refill shelves
def shelf_refill_process(env,name,shelf,num_on_shelf):
    to_refill =  num_on_shelf - shelf['container'].level
    
    yield shelf['container'].put(to_refill)
    
        

def run_sim(num_sim,SIM_TIME,warmup_time,monitor_time,lambd_list,num_rows,num_cols,type_store_widget,outlet_widget,layout_widget, num_cashier):
    if outlet_widget.value in parameter_dict[type_store_widget.value]:
        
        sim_shelves_queue = []
        sim_shelves_user =[]
        sim_refill_count = []
        monitor_period = []
        sim_cashier_queue = []
        

        #each round
        waiting_time_for_shelf = []
        waiting_time_for_cashier = []

        #shelf capacity and number of items on shelf
        shelf_cap = 100
        num_on_shelf = 50
           
        num_reps = 10
        for num in range(num_sim):
            shelves_queue = np.zeros((num_rows, num_cols), dtype=int)
            shelves_user = np.zeros((num_rows, num_cols), dtype=int)
            refill_count = np.zeros((num_rows, num_cols), dtype=int)
            tot_count = {f"Time {i}": np.zeros((num_rows, num_cols), dtype=int) for i in range(int((SIM_TIME)/monitor_time))}
            cust_result[f'period_{num}'] = {}
            display (widgets.HTML(f'<h1>Start running for period {num}</h1>'))
            for reps in range(num_reps):
                print(f"replication {reps}")
                start_time = time.time()
                #print("num =", num)
                #lambd = 1/2
                lambd = lambd_list[num]
                #print("lamdb =", lambd)
                random.seed(reps)
                count = {f"Time {i}": [[0 for _ in range(num_cols)] for _ in range(num_rows)] for i in range(int((SIM_TIME)/monitor_time))}
                count["refill_count"] = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
                #tot_count = {f"Time {i}": np.zeros((num_rows, num_cols), dtype=int) for i in range(int((SIM_TIME)/monitor_time))}
                env = simpy.Environment()

                cashiers = simpy.Resource(env=env, capacity = num_cashier)
                shelves_layout = shelves_location(layout=layout_widget.value, num_rows=num_rows, num_cols=num_cols)[0]
                shelves = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
                #grid = [[simpy.Resource(env=env, capacity=1) for _ in range(num_cols)] for _ in range(num_rows)]

                for pos in shelves_layout:
                    x, y = pos
                    shelves[x][y] = {'resource': simpy.Resource(env=env, capacity=1),
                                     'container': simpy.Container(env=env, capacity=shelf_cap, init=num_on_shelf)
                                    }


                r_matrix = np.array([
                [  1, 0.86, 0.26],
                [ 0.86,  1,  0.13],
                [ 0.26,  0.13,  1]
                ])


                #shopping time parameter
                mean_s = parameter_dict[type_store_widget.value][outlet_widget.value][0]
                sigma_s = parameter_dict[type_store_widget.value][outlet_widget.value][1]

                #basket size parameter
                mean_b = parameter_dict[type_store_widget.value][outlet_widget.value][2]
                sigma_b = parameter_dict[type_store_widget.value][outlet_widget.value][3]

                #proportion of area visited parameter
                mean_a = parameter_dict[type_store_widget.value][outlet_widget.value][4]
                sigma_a = parameter_dict[type_store_widget.value][outlet_widget.value][5]

                size = 100000

                x = correlated_samples(r_matrix=r_matrix,
                                       mean_s=mean_s,
                                       sigma_s=sigma_s,
                                               mean_b=mean_b,
                                               sigma_b=sigma_b,
                                               mean_a=mean_a,
                                               sigma_a=sigma_a,
                                               size=size)
                t = bs.cor(x,bs.Spearman)

                while not np.all(np.abs(t - r_matrix)/(r_matrix) * 100 <= 3):

                    x = correlated_samples(r_matrix=r_matrix,
                                               mean_s=mean_s,
                                               sigma_s=sigma_s,
                                               mean_b=mean_b,
                                               sigma_b=sigma_b,
                                               mean_a=mean_a,
                                               sigma_a=sigma_a,
                                               size=size)
                    t = bs.cor(x,bs.Spearman)
                    #print("hello")

                #print(t)

   
                x_s = [x[i][0] for i in range(size)]
                x_b = [x[i][1] for i in range(size)]
                x_a = [x[i][2] for i in range(size)]  



                env.process(customer_generator(env=env,
                                                   lambd=lambd,
                                                   x=x,
                                                   shelves=shelves,
                                                   cashiers=cashiers,
                                                   num_cols=num_cols,
                                                   num_rows=num_rows,
                                                   num_on_shelf=num_on_shelf,
                                                   count=count,
                                                   warmup_time=warmup_time,
                                                   shelves_layout=shelves_layout,
                                                   layout_widget=layout_widget,
                                                   SIM_TIME = SIM_TIME,
                                                   monitor_time=monitor_time))


                env.run(until=SIM_TIME)
                
                global cust_dict
                global customers_queue_for_cashier
                global customers_queue_for_shelf
                global customers_users_on_shelf
                #global cust_arrival_at_cashier
                #print('period =', num, 'rep =', i)

#                 plt.figure(figsize=(5, 5)) 
#                 plt.hist(cust_area_covered, bins=100,alpha = 0.5,density=True, label="ShopperSim")
#                 plt.hist(cust_x_a, bins=100,alpha = 0.5,density=True, label="Sorenson's")
#                 plt.legend()
#                 plt.title("Proportion of Area Covered")
#                 plt.show()
        

                #Collect data

#                 df = pd.DataFrame(zip(cust_inter_arrival_time, cust_area_covered, cust_shopping_time, cust_basket_size), 
#                       columns=["int_time", "x_a", "x_s", "x_b"])
#                 df.to_excel('data_revise.xlsx', index=False)

                for name in cust_dict:
                    for i in range(int((SIM_TIME)/monitor_time)):
                        #print(i)
                        for item in cust_dict[name]["time"]:
                            #print(item)
                            if item[0] == i*monitor_time:
                                if item[0] >=warmup_time:
                                    count[f"Time {i}"][item[1][0]][item[1][1]] += 1

                

                shelves_queue += customers_queue_for_shelf
                shelves_user += customers_users_on_shelf
                refill_count += count['refill_count']
                #print(shelves_user)
                #print(customers_users_on_shelf)
                #sim_cashier_queue.append(customers_queue_for_cashier)
                #print('Shelves queue = ', customers_queue_for_shelf)
    #             print('Refill count = ', count['refill_count'])
                #waiting_time_for_cashier.append([cust_dict[customer]['waiting_time_for_cashier'] for customer in cust_dict if cust_dict[customer]['get_cashier_time'] != 0 ])
                
                count.pop('refill_count')
                #print(math.ceil(warmup_time/monitor_time))

                for i in range(math.ceil(warmup_time/monitor_time)):
                    del count[f'Time {i}']

               
                for key in count:
                    tot_count[key] += count[key]
                
                #print(count)
                #monitor_period.append(count)
                #monitor_period[0][f'Time {math.ceil(warmup_time/monitor_time)}'] = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
                #print('customer movement = ', monitor_period)
                #print("Cashier queue = ", customers_queue_for_cashier)
                #print("Waiting time for cashier = ", waiting_time_for_cashier)
                
                cust_copy = cust_dict.copy()
                for name in cust_copy:
                    cust_copy[name]['area_covered'] = (cust_copy[name]['total_steps']+1) / (num_cols * num_rows)
                    cust_copy[name].pop('grid', None)
                    cust_copy[name].pop('num_seg_covered', None)
                    cust_copy[name].pop('selected_column', None)
                    cust_copy[name].pop('selected_row', None)
                    cust_copy[name].pop('buy_item_per_seg', None)
                    cust_copy[name].pop('buy_time_per_seg', None)
                    cust_copy[name].pop('count_0', None)
                    cust_copy[name].pop('time', None)
                    cust_copy[name].pop('total_steps', None)
                    cust_copy[name].pop('last_position', None)

                #print(cust_copy)                  
                #print(f"rep_{reps}")
                cust_result[f'period_{num}'][f'rep_{reps}'] = cust_copy
                customers_queue_for_cashier = []
                #customers_queue_for_shelf = []
                cust_dict = {}
                #cust_arrival_at_cashier = []
                end_time = time.time()
                #print("cycle time =", end_time-start_time)
                
             
            for key in tot_count:
                tot_count[key] = np.floor(tot_count[key]/num_reps).astype(int)
            monitor_period.append(tot_count)
            #print(monitor_period)
            #print(np.round(shelves_queue/num_reps).astype(int))
            sim_shelves_queue.append(np.round(shelves_queue/num_reps).astype(int))
            sim_shelves_user.append(np.round(shelves_user/num_reps).astype(int))
            sim_refill_count.append(np.round(refill_count/num_reps).astype(int))
            
        
            display (widgets.HTML(f'<h1>Finish running period {num}</h1>'))
        
        display (widgets.HTML(f'<h1>Preparing statistics...</h1>'))
        # Prepare the data for DataFrame
        data = []
        for period_key, period_val in cust_result.items():
            period_num = int(period_key.split('_')[1])
            for rep_key, rep_val in period_val.items():
                rep_num = int(rep_key.split('_')[1])
                for cust_key, cust_val in rep_val.items():
                    customer_num = int(cust_key.split(' ')[1])
                    flat_dict = {'period': period_num, 'rep': rep_num, 'customer': customer_num}
                    flat_dict.update(cust_val)
                    data.append(flat_dict)

        # Create DataFrame from the list of dictionaries
        df = pd.DataFrame(data)
        
        df.to_excel('statistic_result.xlsx', index=False)
        
        display (widgets.HTML(f'<h1>Statistics completed</h1>'))
            
            #monitor_period = round(np.mean(monitor_period))
        return monitor_period, sim_shelves_queue, sim_refill_count, sim_shelves_user
    else:
        raise ValueError("Error! Choose a new outlet.")
    
    


# In[79]:


#2d
customers_queue_for_shelf = []
customers_users_on_shelf = []

#1d
customers_queue_for_cashier = []

#dict
cust_dict = {}
cust_result = {}
cust_queue_for_cashier = {}

def update_outlet_options(change):
    return  change['new']


def widget_fn():
    title = widgets.VBox([
        widgets.HTML('<h1>Welcome to ShopperSim</h1>'),
        widgets.HTML('<hr style="border: 1px solid #000; margin-top: 5px; margin-bottom: 10px;">')
    ])

    #Type of Store                
    label_1 = widgets.Label('Select type of grocery store: ')
    type_store_widget = widgets.Dropdown(options=type_grocery,
                                         value='Convenience',
                                         disabled=False,
                                         layout=widgets.Layout(width='12%')
                                         )

    label_2 = widgets.Label('Outlet: ')

    outlet_widget = widgets.Dropdown(options=outlet,
                                     value='None',
                                     disabled=False,
                                     layout=widgets.Layout(width='10%')
                                     )

    #Store Layout
    label_3 = widgets.Label('Store Layout: (Number of Aisle x Number of Subcategory)')
    num_col_widget = widgets.IntText(
                                     value=25,
                                     disabled=False,
                                     layout=widgets.Layout(width='5%')
                                    )
    label_4 = widgets.Label('x')
    num_row_widget = widgets.IntText(
                                     value=25,
                                     disabled=False,
                                     layout=widgets.Layout(width='5%')
                                    )
    label_5 = widgets.Label('Type of Layout: ')
    layout_widget = widgets.Dropdown(options=layout_type,
                                     value='North',
                                     disabled=False,
                                     layout=widgets.Layout(width='10%')
                                     )
    label_6 = widgets.Label('Number of Cashiers: ')
    num_cashier_widget = widgets.IntText(
                                     value=1,
                                     disabled=False,
                                     layout=widgets.Layout(width='5%'),
                                     min =1)

    time_intervals = [
        ("From 8.00 to 10.00, Customer arrives with rate:", 0.5),
        ("From 10.00 to 12.00, Customer arrives with rate:", 1),
#         ("From 12.00 to 14.00, Customer arrives with rate:", 2)
    #     ("From 14.00 to 16.00, Customer arrives with rate:", 1),
    #     ("From 16.00 to 18.00, Customer arrives with rate:", 0.01),
    #     ("From 18.00 to 20.00, Customer arrives with rate:", 3),
    #     ("From 20.00 to 22.00, Customer arrives with rate:", 5)

    ]

    #     ("From 11.00 to 12.00, Customer arrives every:", 1),
    #     ("From 12.00 to 13.00, Customer arrives every:", 1),
    #     ("From 13.00 to 14.00, Customer arrives every:", 1)

    # ("From 14.00 to 15.00, Customer arrives every:", 1),
    #     ("From 15.00 to 16.00, Customer arrives every:", 1),
    #     ("From 16.00 to 17.00, Customer arrives every:", 1),
    #     ("From 17.00 to 18.00, Customer arrives every:", 1),
    #     ("From 18.00 to 19.00, Customer arrives every:", 1),
    #     ("From 19.00 to 20.00, Customer arrives every:", 1),
    #     ("From 20.00 to 21.00, Customer arrives every:", 1)
    # List to store the created widgets
    widgets_list = []
    minute_label = widgets.Label("customers/min")
    # Loop through the time intervals and create the corresponding widgets
    for label_text, default_value in time_intervals:
        label = widgets.Label(label_text)
        float_text = widgets.FloatText(
            value=default_value,
            disabled=False,
            layout=widgets.Layout(width='5%')
        )
        widgets_list.append(widgets.HBox([label, float_text, minute_label]))

    # Create a VBox to display all the widgets
    automated_widgets = widgets.VBox(widgets_list)


    note = widgets.VBox([
            widgets.HTML('<h1 style="font-size: 20px;">Users Notes (Read First):</h1>'),
            widgets.Label("US Super has only outlet Store A to K, UK Super has only outlet Store A, US Hyper has only outlet Store A and B"),
            widgets.Label("AU Liquor has only outlet Store A to L, AU Drug has only outlet Store A to G, and Convinience has only outlet None"),
            widgets.Label("Customer's rate must be positive number, Number of cashiers must be positive integer"),

            widgets.HTML('<hr style="border: 1px solid #000; margin-top: 5px; margin-bottom: 10px;">')

    ])




    #Observe change
    type_store_widget.observe(update_outlet_options, names='value')
    outlet_widget.observe(update_outlet_options, names='value')
    num_col_widget.observe(update_outlet_options, names='value')
    num_row_widget.observe(update_outlet_options, names='value')
    layout_widget.observe(update_outlet_options, names='value')
    num_cashier_widget.observe(update_outlet_options, names='value')

    for i in range(len(automated_widgets.children)):
        automated_widgets.children[i].children[1].observe(update_outlet_options, names='value')
    # reverse_lambd_1_widget.observe(update_outlet_options, names='value')
    # reverse_lambd_2_widget.observe(update_outlet_options, names='value')
    # reverse_lambd_3_widget.observe(update_outlet_options, names='value')
    return num_col_widget,num_row_widget,automated_widgets,num_cashier_widget,type_store_widget,outlet_widget,layout_widget,title,label_1,label_2,label_3,label_4,label_5,label_6,note 


#If there are clicks on botton it will add constraint to the optimization problem and then optimize it
def on_botton_change(change):
    
    # store layout
    num_cols = widget_fn()[0].value
    num_rows = widget_fn()[1].value
    

    SIM_TIME = 120 #Sim time for 60 mins in each simulation
    warmup_time = 0  #first 20 min must be removed, since it is not stable yet

    #monitor shelves using
    monitor_time = 5 #Monitor every ... min
    lambd_list = [widget_fn()[2].children[i].children[1].value for i in range(len(widget_fn()[2].children))]
    #print(lambd_list)
    #lambd_list = [1/reverse_lambd_1_widget.value,1/reverse_lambd_2_widget.value,1/reverse_lambd_3_widget.value]
    #print(lambd_list)
    num_sim = len(lambd_list)
    
    num_cashier = widget_fn()[3].value
    
    #print("num_sim =", num_sim)
    
    result = run_sim(num_sim=num_sim,
                SIM_TIME=SIM_TIME,
                warmup_time=warmup_time,
                monitor_time=monitor_time,
                lambd_list=lambd_list,
                num_rows=num_rows,
                num_cols=num_cols,
                type_store_widget=widget_fn()[4],
                outlet_widget=widget_fn()[5],
                layout_widget=widget_fn()[6],
                num_cashier=num_cashier)
#     for item in result[0]:
#         print('Cust movement = ', item)
    #print(len(result[0]))
    #print('Shelves queue = ', result[1])
#     print(len(result[1]))
    #print('Refill count = ', result[2])
#     print(len(result[2]))
#     print("Cashier queue = ", result[3][0])
#     print(len(result[3]))

    def plot(heatmap,shelf_queue,shelf_refill,shelf_user,n,monitor_time):
        fig, ax = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)
        #ax[0, 0].set_ylabel("Subcategory")
        #ax[0, 0].set_xlabel("Number of Aisle")

        sm = ScalarMappable(cmap='Greens',norm=plt.Normalize(vmax=30))
        sm.set_array([])

        def animate(frame):
            im = ax[0, 0].imshow(heatmap[frame],cmap='Greens', vmax = 30)
            if frame*monitor_time < 10:
                ax[0, 0].set_title('Customer Movement at time '+str(8 + 2*n)+'.0' + str(frame*monitor_time)+':')
            elif frame*monitor_time >= 10 and frame*monitor_time < 60:
                ax[0, 0].set_title('Customer Movement at time '+str(8 + 2*n)+'.' + str(frame*monitor_time)+':')
            elif frame*monitor_time >= 60 and frame*monitor_time < 70:
                ax[0, 0].set_title('Customer Movement at time '+str(8 + 2*n+1)+'.0' + str((frame*monitor_time)-60)+':')
            elif frame*monitor_time >= 70 and frame*monitor_time < 120:
                ax[0, 0].set_title('Customer Movement at time '+str(8 + 2*n+1)+'.' + str((frame*monitor_time)-60)+':')
            sm.set_array(heatmap[frame])
            for txt in ax[0, 0].texts:
                txt.set_visible(False)
            for i in range(len(heatmap[frame])):
                for j in range(len(heatmap[frame][0])):
                    text = ax[0, 0].text(j, i, str(heatmap[frame][i][j]), ha='center', va='center', color='black', fontsize=8)
            return im
            


        play = Play(value=0, min=0, max=len(heatmap) - 1, step=1, interval=1500)
        #display(interactive(animate, frame=play))
        display(interact(animate, frame=play))
        
        # Queue at shelf over time
        ax[0, 1].imshow(shelf_queue, cmap=cm.gray, vmin=-20, vmax=5)  
        for i in range(len(shelf_queue)):
            for j in range(len(shelf_queue[0])):
                text = ax[0, 1].text(j, i, str(shelf_queue[i][j]), ha='center', va='center', color='black', fontsize=8)

        ax[0, 1].set_title("Shelves Queue Map")
        #ax[0, 1].set_ylabel("S")
        #ax[0, 1].set_xlabel("Number of Aisle")
        
         # Number of refill at shelf result
        ax[1, 0].imshow(shelf_refill, cmap='coolwarm', vmin=-2, vmax=13)

        for i in range(len(shelf_refill)):
            for j in range(len(shelf_refill[0])):
                text = ax[1, 0].text(j, i, str(shelf_refill[i][j]), ha='center', va='center', color='black', fontsize=8)

        ax[1, 0].set_title("Shelves Refill Map")
        #ax[1, 0].set_ylabel("Subcategory")
        #ax[1, 0].set_xlabel("Number of Aisle")
        
        
        # Users at shelf over time
        ax[1, 1].imshow(shelf_user, cmap='PuBu', vmin=0, vmax=5)  
        for i in range(len(shelf_user)):
            for j in range(len(shelf_user[0])):
                text = ax[1, 1].text(j, i, str(shelf_user[i][j]), ha='center', va='center', color='black', fontsize=8)

        ax[1, 1].set_title("Shelves User Map")
        #ax[1, 0].set_ylabel("Subcategory")
        #ax[1, 0].set_xlabel("Number of Aisle")
     
    #print("n =", n)
    heatmap = [[result[0][n][f"Time {i}"] for i in range(math.ceil(warmup_time/monitor_time), int(SIM_TIME/monitor_time))] for n in range(num_sim)]
    
    #print("result1n = ", result[1][0])
    shelf_queue = [result[1][n] for n in range(num_sim)]
    shelf_refill = [result[2][n] for n in range(num_sim)]
    shelf_user = [result[3][n] for n in range(num_sim)]
    
    for n in range(num_sim):
        plot(heatmap=heatmap[n],
             shelf_queue=shelf_queue[n],
             shelf_refill=shelf_refill[n],
             shelf_user=shelf_user[n],
             n=n,
             monitor_time=monitor_time)
   
  

# Attach the event handler to the Optimize button.
def run_shoppersim():
    run_sim_botton = widgets.ToggleButton(value=False,description='Run ShopperSim',disabled=False,
                                          button_style='info', # ['success', 'info', 'warning', 'danger' or '']
                                          tooltip='Description'# icon='check' # (FontAwesome names without the `fa-` prefix)
                                          )
    run_sim_botton.observe(lambda change: on_botton_change(change), names='value')        
    display(widgets.VBox([
                         widget_fn()[7],
                         widget_fn()[14],
                         widgets.HBox([widget_fn()[8], widget_fn()[4], widget_fn()[9], widget_fn()[5]]),
                         widgets.HBox([widget_fn()[10], widget_fn()[0], widget_fn()[11], widget_fn()[1]]),
                         widgets.HBox([widget_fn()[12], widget_fn()[6]]),
                         widgets.HBox([ widget_fn()[13], widget_fn()[3]]),
                         widget_fn()[2],
                         widgets.HBox([run_sim_botton], layout=widgets.Layout(justify_content='flex-end', margin='20px 0'))
                        ]))


# In[80]:


# #2d
# customers_queue_for_shelf = []
# customers_users_on_shelf = []

# #1d
# customers_queue_for_cashier = []

# #dict
# cust_dict = {}
# cust_result = {}
# cust_queue_for_cashier = {}

# def update_outlet_options(change):
#     return  change['new']


# title = widgets.VBox([
#     widgets.HTML('<h1>Welcome to ShopperSim</h1>'),
#     widgets.HTML('<hr style="border: 1px solid #000; margin-top: 5px; margin-bottom: 10px;">')
# ])

# #Type of Store                
# label_1 = widgets.Label('Select type of grocery store: ')
# type_store_widget = widgets.Dropdown(options=type_grocery,
#                                      value='Convenience',
#                                      disabled=False,
#                                      layout=widgets.Layout(width='12%')
#                                      )

# label_2 = widgets.Label('Outlet: ')

# outlet_widget = widgets.Dropdown(options=outlet,
#                                  value='None',
#                                  disabled=False,
#                                  layout=widgets.Layout(width='10%')
#                                  )

# #Store Layout
# label_3 = widgets.Label('Store Layout: (Number of Aisle x Number of Subcategory)')
# num_col_widget = widgets.IntText(
#                                  value=25,
#                                  disabled=False,
#                                  layout=widgets.Layout(width='5%')
#                                 )
# label_4 = widgets.Label('x')
# num_row_widget = widgets.IntText(
#                                  value=25,
#                                  disabled=False,
#                                  layout=widgets.Layout(width='5%')
#                                 )
# label_5 = widgets.Label('Type of Layout: ')
# layout_widget = widgets.Dropdown(options=layout_type,
#                                  value='North',
#                                  disabled=False,
#                                  layout=widgets.Layout(width='10%')
#                                  )
# label_6 = widgets.Label('Number of Cashiers: ')
# num_cashier_widget = widgets.IntText(
#                                  value=1,
#                                  disabled=False,
#                                  layout=widgets.Layout(width='5%'),
#                                  min =1)

# time_intervals = [
#     ("From 8.00 to 10.00, Customer arrives with rate:", 2),
#     ("From 10.00 to 12.00, Customer arrives with rate:", 3),
#     ("From 12.00 to 14.00, Customer arrives with rate:", 0.2),
#     ("From 14.00 to 16.00, Customer arrives with rate:", 5),
#     ("From 16.00 to 18.00, Customer arrives with rate:", 0.8),
#     ("From 18.00 to 20.00, Customer arrives with rate:", 3),
#     ("From 20.00 to 22.00, Customer arrives with rate:", 5)
    
# ]

# #     ("From 11.00 to 12.00, Customer arrives every:", 1),
# #     ("From 12.00 to 13.00, Customer arrives every:", 1),
# #     ("From 13.00 to 14.00, Customer arrives every:", 1)
    
# # ("From 14.00 to 15.00, Customer arrives every:", 1),
# #     ("From 15.00 to 16.00, Customer arrives every:", 1),
# #     ("From 16.00 to 17.00, Customer arrives every:", 1),
# #     ("From 17.00 to 18.00, Customer arrives every:", 1),
# #     ("From 18.00 to 19.00, Customer arrives every:", 1),
# #     ("From 19.00 to 20.00, Customer arrives every:", 1),
# #     ("From 20.00 to 21.00, Customer arrives every:", 1)
# # List to store the created widgets
# widgets_list = []
# minute_label = widgets.Label("customers/min")
# # Loop through the time intervals and create the corresponding widgets
# for label_text, default_value in time_intervals:
#     label = widgets.Label(label_text)
#     float_text = widgets.FloatText(
#         value=default_value,
#         disabled=False,
#         layout=widgets.Layout(width='5%')
#     )
#     widgets_list.append(widgets.HBox([label, float_text, minute_label]))

# # Create a VBox to display all the widgets
# automated_widgets = widgets.VBox(widgets_list)


# note = widgets.VBox([
#         widgets.HTML('<h1 style="font-size: 20px;">Users Notes (Read First):</h1>'),
#         widgets.Label("US Super has only outlet Store A to K, UK Super has only outlet Store A, US Hyper has only outlet Store A and B"),
#         widgets.Label("AU Liquor has only outlet Store A to L, AU Drug has only outlet Store A to G, and Convinience has only outlet None"),
#         widgets.Label("Customer's rate must be positive number, Number of cashiers must be positive integer"),

#         widgets.HTML('<hr style="border: 1px solid #000; margin-top: 5px; margin-bottom: 10px;">')
    
# ])




# run_sim_botton = widgets.ToggleButton(value=False,description='Run ShopperSim',disabled=False,
#                                       button_style='info', # ['success', 'info', 'warning', 'danger' or '']
#                                       tooltip='Description'# icon='check' # (FontAwesome names without the `fa-` prefix)
#                                       )


# #Observe change
# type_store_widget.observe(update_outlet_options, names='value')
# outlet_widget.observe(update_outlet_options, names='value')
# num_col_widget.observe(update_outlet_options, names='value')
# num_row_widget.observe(update_outlet_options, names='value')
# layout_widget.observe(update_outlet_options, names='value')
# num_cashier_widget.observe(update_outlet_options, names='value')

# for i in range(len(automated_widgets.children)):
#     automated_widgets.children[i].children[1].observe(update_outlet_options, names='value')
# # reverse_lambd_1_widget.observe(update_outlet_options, names='value')
# # reverse_lambd_2_widget.observe(update_outlet_options, names='value')
# # reverse_lambd_3_widget.observe(update_outlet_options, names='value')



# #If there are clicks on botton it will add constraint to the optimization problem and then optimize it
# def on_botton_change(change):
    
#     # store layout
#     num_cols = num_col_widget.value
#     num_rows = num_row_widget.value
    

#     SIM_TIME = 120 #Sim time for 60 mins in each simulation
#     warmup_time = 0  #first 20 min must be removed, since it is not stable yet

#     #monitor shelves using
#     monitor_time = 5 #Monitor every ... min
#     lambd_list = [automated_widgets.children[i].children[1].value for i in range(len(automated_widgets.children))]
#     #print(lambd_list)
#     #lambd_list = [1/reverse_lambd_1_widget.value,1/reverse_lambd_2_widget.value,1/reverse_lambd_3_widget.value]
#     #print(lambd_list)
#     num_sim = len(lambd_list)
    
#     num_cashier = num_cashier_widget.value
    
#     #print("num_sim =", num_sim)
    
#     result = run_sim(num_sim=num_sim,
#                 SIM_TIME=SIM_TIME,
#                 warmup_time=warmup_time,
#                 monitor_time=monitor_time,
#                 lambd_list=lambd_list,
#                 num_rows=num_rows,
#                 num_cols=num_cols,
#                 type_store_widget=type_store_widget,
#                 outlet_widget=outlet_widget,
#                 layout_widget=layout_widget,
#                 num_cashier=num_cashier)
#     #for item in result[0]:
#       #  print('Cust movement = ', item)
#     #print(len(result[0]))
#     #print('Shelves queue = ', result[1])
# #     print(len(result[1]))
#     #print('Refill count = ', result[2])
# #     print(len(result[2]))
# #     print("Cashier queue = ", result[3][0])
# #     print(len(result[3]))

#     def plot(heatmap,shelf_queue,shelf_refill,shelf_user,n,monitor_time):
#         fig, ax = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)
#         #ax[0, 0].set_ylabel("Subcategory")
#         #ax[0, 0].set_xlabel("Number of Aisle")

#         sm = ScalarMappable(cmap='Greens',norm=plt.Normalize(vmax=30))
#         sm.set_array([])

#         def animate(frame):
#             im = ax[0, 0].imshow(heatmap[frame],cmap='Greens', vmax = 30)
#             if frame*monitor_time < 10:
#                 ax[0, 0].set_title('Customer Movement at time '+str(8 + 2*n)+'.0' + str(frame*monitor_time)+':')
#             elif frame*monitor_time >= 10 and frame*monitor_time < 60:
#                 ax[0, 0].set_title('Customer Movement at time '+str(8 + 2*n)+'.' + str(frame*monitor_time)+':')
#             elif frame*monitor_time >= 60 and frame*monitor_time < 70:
#                 ax[0, 0].set_title('Customer Movement at time '+str(8 + 2*n+1)+'.0' + str((frame*monitor_time)-60)+':')
#             elif frame*monitor_time >= 70 and frame*monitor_time < 120:
#                 ax[0, 0].set_title('Customer Movement at time '+str(8 + 2*n+1)+'.' + str((frame*monitor_time)-60)+':')
#             sm.set_array(heatmap[frame])
#             for txt in ax[0, 0].texts:
#                 txt.set_visible(False)
#             for i in range(len(heatmap[frame])):
#                 for j in range(len(heatmap[frame][0])):
#                     text = ax[0, 0].text(j, i, str(heatmap[frame][i][j]), ha='center', va='center', color='black', fontsize=8)
#             return im
            


#         play = Play(value=0, min=0, max=len(heatmap) - 1, step=1, interval=1500)
#         #display(interactive(animate, frame=play))
#         display(interact(animate, frame=play))
        
#         # Queue at shelf over time
#         ax[0, 1].imshow(shelf_queue, cmap=cm.gray, vmin=-20, vmax=5)  
#         for i in range(len(shelf_queue)):
#             for j in range(len(shelf_queue[0])):
#                 text = ax[0, 1].text(j, i, str(shelf_queue[i][j]), ha='center', va='center', color='black', fontsize=8)

#         ax[0, 1].set_title("Shelves Queue Map")
#         #ax[0, 1].set_ylabel("S")
#         #ax[0, 1].set_xlabel("Number of Aisle")
        
#          # Number of refill at shelf result
#         ax[1, 0].imshow(shelf_refill, cmap='coolwarm', vmin=-2, vmax=13)

#         for i in range(len(shelf_refill)):
#             for j in range(len(shelf_refill[0])):
#                 text = ax[1, 0].text(j, i, str(shelf_refill[i][j]), ha='center', va='center', color='black', fontsize=8)

#         ax[1, 0].set_title("Shelves Refill Map")
#         #ax[1, 0].set_ylabel("Subcategory")
#         #ax[1, 0].set_xlabel("Number of Aisle")
        
        
#         # Users at shelf over time
#         ax[1, 1].imshow(shelf_user, cmap='PuBu', vmin=0, vmax=5)  
#         for i in range(len(shelf_user)):
#             for j in range(len(shelf_user[0])):
#                 text = ax[1, 1].text(j, i, str(shelf_user[i][j]), ha='center', va='center', color='black', fontsize=8)

#         ax[1, 1].set_title("Shelves User Map")
#         #ax[1, 0].set_ylabel("Subcategory")
#         #ax[1, 0].set_xlabel("Number of Aisle")
     
#     #print("n =", n)
#     heatmap = [[result[0][n][f"Time {i}"] for i in range(math.ceil(warmup_time/monitor_time), int(SIM_TIME/monitor_time))] for n in range(num_sim)]
    
#     #print("result1n = ", result[1][0])
#     shelf_queue = [result[1][n] for n in range(num_sim)]
#     shelf_refill = [result[2][n] for n in range(num_sim)]
#     shelf_user = [result[3][n] for n in range(num_sim)]
    
#     for n in range(num_sim):
#         plot(heatmap=heatmap[n],
#              shelf_queue=shelf_queue[n],
#              shelf_refill=shelf_refill[n],
#              shelf_user=shelf_user[n],
#              n=n,
#              monitor_time=monitor_time)
   
  

# # Attach the event handler to the Optimize button.

# run_sim_botton.observe(lambda change: on_botton_change(change), names='value')        
# display(widgets.VBox([
#                      title,
#                      note,
#                      widgets.HBox([label_1, type_store_widget, label_2, outlet_widget]),
#                      widgets.HBox([label_3, num_col_widget, label_4, num_row_widget]),
#                      widgets.HBox([label_5, layout_widget]),
#                      widgets.HBox([label_6, num_cashier_widget]),
#                      automated_widgets,
#                      widgets.HBox([run_sim_botton], layout=widgets.Layout(justify_content='flex-end', margin='20px 0'))
#                     ]))


# In[ ]:




