import numpy as np
from stl import mesh
import matplotlib.pyplot as plt

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

solid_body = mesh.Mesh.from_file('School Building.stl')

#print(solid_body.vectors)

#print(solid_body.vectors)

#print(len(solid_body.vectors))
boundry = []
for triangles in solid_body.vectors:
    #print("Triangles", triangles)
    for points in triangles:
        #print(points[2])
        #for zvalue in points:
         #   print(zvalue)
        if points[2] <= 0.8:
           #this triangle is in layer keep move to next
           # only need x and y data
           if [points[0],points[1]] not in boundry:
                boundry.append([points[0],points[1]]) 
        
#print(boundry)

points = boundry
plt.figure()

for i in range(len(points)):
    loc = Point(points[i][0],points[i][1])
    plt.scatter(loc.x, loc.y)
    print('X' + str(loc.x) +' Y' + str(loc.y))
plt.grid(True)
plt.show()