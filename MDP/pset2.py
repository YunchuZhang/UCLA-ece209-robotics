import numpy as np
L = 6
W = 6
# 12 heading
h = [0,1,2,3,4,5,6,7,8,9,10,11]
# action = rotation (left, right) firts with Pe + move (foward,backward)
# or only move  1-2Pe not rotate
# or stay still 
#      11,0,1
#8,9,10      2,3,4
#       5,6,7
# turn left h+1 right h-1 mod%12
S = 