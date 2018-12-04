import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
from scipy.spatial.distance import cdist


input_image = cv2.imread('Input_Images/Enrico.jpg')
target_image = cv2.imread('Target_Images/Benedict_Cumberbatch.jpg')


neighborhood = 5
p = 0.2
m=1
iterations  = 1

i_height = input_image.shape[0]
i_width = input_image.shape[1]
i_depth = input_image.shape[2]
o_height = target_image.shape[0] 
o_width = target_image.shape[1]
o_depth = target_image.shape[2]
n_2 = int(math.floor(neighborhood/2))
	
input_image_gray = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY).astype("float32")
target_image_gray = cv2.cvtColor(target_image,cv2.COLOR_BGR2GRAY).astype("float32")

input_image_gray = cv2.normalize(input_image_gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
target_image_gray = cv2.normalize(target_image_gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

output_image = np.zeros((o_height+n_2, o_width+n_2*2, o_depth))
used_heights = np.zeros((o_height+n_2, o_width+n_2*2))
used_widths = np.zeros((o_height+n_2, o_width+n_2*2))

used_heights[0:n_2-1,:] = np.round(np.random.rand(n_2-1, o_width+n_2*2)*(i_height-2)+1)
used_widths[0:n_2-1,:] = np.round(np.random.rand(n_2-1, o_width+n_2*2)*(i_width-2)+1)
used_heights[:,0:n_2-1] = np.round(np.random.rand(o_height+n_2, n_2-1)*(i_height-2)+1)
used_widths[:,0:n_2-1] = np.round(np.random.rand(o_height+n_2, n_2-1)*(i_width-2)+1)
used_heights[:,n_2+o_width:n_2*2+o_width-1] = np.round(np.random.rand(o_height+n_2, n_2-1)*(i_height-2)+1)
used_widths[:,n_2+o_width:n_2*2+o_width-1] = np.round(np.random.rand(o_height+n_2, n_2-1)*(i_width-2)+1)

for h in range(0,n_2):
    for w in range(0,o_width+n_2*2):
        output_image[h,w,:] = input_image[int(used_heights[h,w]),int(used_widths[h,w]),:]

for h in range(n_2-1,o_height+n_2):
    for w in np.array(np.hstack((np.arange(0,n_2),np.arange(n_2+o_width,n_2*2+o_width)))):
        output_image[h-1,w-1,:] = input_image[int(used_heights[h-1,w-1]),int(used_widths[h-1,w-1]),:]

for iteration in range(0,iterations):
    print iteration
    
    for h in range(n_2,o_height+n_2):
        
        for w in range(n_2,o_width+n_2):
            print h, w
            candidate_locations = []
            candidate_pixels = {}
            count=1
            search_height = np.arange(0,n_2+1,1)
            search_width = np.arange(0,neighborhood,1)
            
            if iteration>1:
                search_height = np.arange(-n_2,n_2+1,1)
            
            for c_h in range(len(search_height)):
                for c_w in range(len(search_width)):
                    c_w_adj = c_w-n_2
                    if ((c_h>0 or (c_h==0 and c_w_adj)) or iteration>1) and (h-c_h <= o_height+n_2):
                        new_height = int(used_heights[h-c_h-1,w+c_w_adj-1]+c_h);
                        new_width = int(used_widths[h-c_h-1,w+c_w_adj-1]-c_w_adj);
                        
                        while ((new_height < neighborhood) or (new_height > i_height-neighborhood)) or ((new_width < neighborhood)or (new_width > i_width-neighborhood)):
                            new_height = int(np.round(np.random.rand(1)*(i_height-2)+1)[0])
                            new_width = int(np.round(np.random.rand(1)*(i_width-2)+1)[0])
                        
                        #print candidate_locations
                        #candidate_locations = np.resize(candidate_locations,(new_height,new_width))
                        list1 = [new_height,new_width]
                        candidate_locations.append(list1)
                        candidate_pixels[count] = input_image[new_height-1,new_width-1,:]
                        count=count+1
            
            if np.random.rand()<p:
                new_height = int(np.round(np.random.rand(1)*(i_height-2)+1)[0])
                new_width = int(np.round(np.random.rand(1)*(i_width-2)+1)[0])
            
                while ((new_height < neighborhood) or (new_height > i_height-neighborhood)) or((new_width < neighborhood) or  (new_width > i_width-neighborhood)):
                    new_height = int(np.round(np.random.rand(1)*(i_height-2)+1)[0])
                    new_width = int(np.round(np.random.rand(1)*(i_width-2)+1)[0])
                
                #candidate_locations = np.reshape(candidate_locations,(new_height,new_width))
                list1 = [new_height,new_width]
                candidate_locations.append(list1)
                candidate_pixels[count] = input_image[new_height-1,new_width-1,:]
            
            [C,unique_indices] = np.unique(candidate_locations, 'rows')
            
            best_dist = 10000
            best_pixel = []
            best_location = []
            
            for i in (unique_indices):
                #print i
                #print candidate_locations[i%len(candidate_locations)]
                #print candidate_locations[i%len(candidate_locations)]
                c_h = candidate_locations[i%len(candidate_locations)][0]
                c_w = candidate_locations[i%len(candidate_locations)][1]
                
                if iteration > 1:
                    print "hi"
                    
                else:
                    #print "ch"
                    #print c_h
                    #print "cw"
                    #print c_w
                    l = np.reshape( input_image[c_h-n_2-1:c_h-1,c_w-n_2-1:c_w+n_2,:],(1,3*neighborhood*n_2))
                    #print l.shape
                    
                    input_values = np.concatenate((np.reshape( input_image[c_h-n_2-1:c_h-1,c_w-n_2-1:c_w+n_2,:],(1,3*neighborhood*n_2)),
                                    np.reshape(input_image[c_h,c_w-n_2-1:c_w-1,:],(1,3*n_2))),axis=1)
                    
                    result_values = np.concatenate((np.reshape(output_image[h-n_2:h,w-n_2:w+n_2+1,:],(1,3*neighborhood*n_2)),
                                    np.reshape(output_image[h,w-n_2:w,:],(1,3*n_2))),axis=1)
                    
                    input_result_distance = cdist(input_values,result_values)
                    n = neighborhood*n_2+n_2
                   
                    height = np.arange(np.maximum(-n_2,1-(h-n_2)),np.minimum(n_2,o_height-(h-n_2))+1)
                    width = np.arange(np.maximum(-n_2,1-(w-n_2)),np.minimum(n_2,o_width-(w-n_2))+1)
                    
                    
                    #input_values = input_image_gray[height+c_h,width+c_w]
                    
                    a = height+c_h
                    b = width+c_w
                    input_values = np.zeros((len(a),len(b)))
                    
                    for j in range(len(a)):
                        for k in range(len(b)):
                            input_values[j,k]=input_image_gray[a[j],b[k]]
                    
                    a = (h-n_2)+height-1
                    b = (w-n_2)+width-1
                    target_values = np.zeros((len(a),len(b)))
                    
                    for j in range(len(a)):
                        for k in range(len(b)):
                            target_values[j,k]=target_image_gray[a[j],b[k]]
                    
                    
                    
                    
                    
                    #target_values = target_image_gray[(h-n_2)+height-1,(w-n_2)+width-1]
                    
                    input_target_distance = math.pow(np.mean(input_values[:]) - np.mean(target_values[:]),2)
                    distance = m*input_target_distance + (1/math.pow(n,2))*input_result_distance
                
                if distance < best_dist:
                    best_pixel = candidate_pixels[i%len(candidate_locations)+1]
                    #print candidate_locations[i%len(candidate_locations)]
                    best_location = candidate_locations[i%len(candidate_locations)]
                    best_dist = distance
            
            output_image[h-1,w-1,:] = best_pixel
            used_heights[h-1,w-1] = best_location[0]
            used_widths[h-1,w-1] = best_location[1]

new_output = output_image[n_2:o_height+n_2-1, n_2:o_width+n_2-1, :]

cv2.imwrite('output.png',new_output)
            
            
        
        
                
            
                        