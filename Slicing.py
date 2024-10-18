import matplotlib.pyplot as plt
import math
import pyvista as pv
import stl
from stl import mesh
import numpy as np
from scipy.spatial import ConvexHull
import random

class Slicer_backend:

    def read_file_show(filepath):
        solid_body = mesh.Mesh.from_file(filepath)
        file = pv.read(filepath)
        return solid_body, file.plot()

    def get_points_per_slice(solid_body):
        boundry = []
        for triangles in solid_body.vectors:
            for points in triangles:
                if points[2] <= 0.8:
                   #this triangle is in layer keep move to next
                   # only need x and y data 
                   if [points[0],points[1]] not in boundry:
                        boundry.append([points[0],points[1]]) 
        return boundry

def random_start(perimeter, prevPoint = None):
    pointList = [(i, j) for i, j in perimeter]
    index = 0
    if prevPoint == None:
        return (0, pointList[0])
    else:
        if index > 0:
            index -= len(pointList) / 2
            return (index, pointList[index])            
        else:
            index += len(pointList) / 2
            return (index, pointList[index])
        
def convert_to_gcode(points):
    gcode_commands = []
    gcode_commands.append("G21 ; Set units to millimeters")
    gcode_commands.append("G90 ; Use absolute positioning")
    gcode_commands.append("G1 F1500 ; Set feedrate")

    for point in points:
        (x_start, y_start) = point
        gcode_commands.append(f"G1 X{x_start:.4f} Y{y_start:.4f}")  # Move to start point

    return gcode_commands


if __name__ == "__main__":
    # Load the 3D object and visualize it
    object, visualization = Slicer_backend.read_file_show(filepath=r"C:\Users\zzcro\Desktop\Lab_Assignments\Keck\TestCylinder.stl")
    points = Slicer_backend.get_points_per_slice(object)
    points.append(points[0])
    plt.figure()
    startPoint = random_start(points, None)
    gCodePath = points
    xPoints = [point[0] for point in points]
    yPoints = [point[1] for point in points]
    plt.scatter(xPoints, yPoints, c='blue')
    plt.plot(xPoints, yPoints)
    plt.scatter(startPoint[1][0], startPoint[1][1], c='red')
    
        #print('X' + str(points[i][0]) +' Y' + str(points[i][1]))
    ''' 
    1.5 Milimeter or 0.8 Milimeter Extrusion Diameter
    0.4 Milimeter Height
    '''
    g_commands = convert_to_gcode(gCodePath) 
    output_file_path = "/Users/zzcro/Desktop/Lab_Assignments/Keck/generated.gcode"
    with open(output_file_path, "w") as gcode_file:
        for command in g_commands:
             gcode_file.write(command + "\n")
     
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()
    
    
    
    
    
    ''' Archived Work
    def slicing_object(obj, layer_height):
        minz = maxz = None
        for p in obj.points:
            # p contains (x, y, z)
            if minz is None:
                minz = p[stl.Dimension.Z]
                maxz = p[stl.Dimension.Z]
            else:
                maxz = max(p[stl.Dimension.Z], maxz)
                minz = min(p[stl.Dimension.Z], minz)

        z_length = (maxz - minz)
        if z_length % layer_height:
            num_slices = (int(z_length / layer_height) + 1)
        else:
            num_slices = int(z_length / layer_height)

        slices = obj.slice_along_axis(n=num_slices, axis="z")
        return slices
    
    def clockwise_sort(points):
        sortedpoints = []
        for i in range(len(points)):
            xpoints, ypoints = points[i]

            # Step 1: Compute the center of all points
            x_center = sum(xpoints) / len(xpoints)
            y_center = sum(ypoints) / len(ypoints)

            # Combine x and y points into a list of tuples
            combined_points = list(zip(xpoints, ypoints))

            # Step 2: Calculate the angle for each point with respect to the center
            def angle_from_center(point):
                x, y = point
                return math.atan2(y - y_center, x - x_center)

            # Step 3: Sort points based on the angle in descending order for clockwise sorting
            sorted_points = sorted(combined_points, key=angle_from_center, reverse=True)

            sortedpoints.append(sorted_points)

        return sortedpoints

    def clockwise_sort_from_start_point(points, start_point):
        start_x, start_y = start_point

        # Define a function to calculate the angle of each point with respect to the start_point
        def angle_from_start(point):
            x, y = point
            # Calculate the angle with respect to the start_point
            return math.atan2(y - start_y, x - start_x)

        # Sort the points based on the angle in descending order for clockwise sorting
        sorted_points = sorted(points, key=angle_from_start, reverse=True)

        return sorted_points

    def find_outer_inner_perimeters(points):
        outer_perimeters = []
        inner_perimeters = []

        for sorted_slice in points:
            combined_points = np.array(sorted_slice)  # Convert list of tuples to a NumPy array

            # Find the convex hull of the set of points
            hull = ConvexHull(combined_points)
            hull_points = combined_points[hull.vertices]

            # Extract outer perimeter points
            outer_perimeter = [(x, y) for x, y in hull_points]
            outer_perimeters.append(outer_perimeter)

            # Find inner perimeter points (those not part of the convex hull)
            inner_points = set(map(tuple, combined_points)) - set(outer_perimeter)
            inner_perimeter = list(inner_points)  # Convert set back to list
            inner_perimeters.append(inner_perimeter)

        return outer_perimeters, inner_perimeters

    def generate_points(points, dist_b_points):

        new_points = points.copy()

        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]

            if x1 == x2:  # Vertical line segment
                num_points = int(abs((y2 - y1) / dist_b_points))
                for j in range(1, num_points):
                    if y1<y2:
                        y = y1 + (j * dist_b_points)
                    else:
                        y = y1 - (j * dist_b_points)
                    candidate_point = (x1, y)
                    new_points.append(candidate_point)

            elif y1 == y2:  # Horizontal line segment
                num_points = int(abs((x2 - x1) / dist_b_points))
                for k in range(1, num_points):
                    if x1<x2:
                        x = x1 + (k * dist_b_points)
                    else:
                        x = x1 - (k * dist_b_points)
                    candidate_point = (x, y1)
                    new_points.append(candidate_point)

            else:  # Diagonal line segment
                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                num_points = int(dist / dist_b_points)
                if num_points > 1:  # Only add intermediate points if necessary
                    for n in range(1, num_points):
                        t = n / num_points
                        x = x1 + t * (x2 - x1)
                        y = y1 + t * (y2 - y1)
                        candidate_point = (x, y)
                        new_points.append(candidate_point)

        return new_points

    def generate_inward_steps(points, step_length):
        # Calculate the center of the geometry
        center_x = sum(x for x, y in points) / len(points)
        center_y = sum(y for x, y in points) / len(points)
        center = (center_x, center_y)

        layers = [points]  # Start with the outermost layer

        while True:
            new_layer = []
            for x, y in layers[-1]:
                # Calculate direction vector from point to center
                direction_x = center_x - x
                direction_y = center_y - y

                # Normalize the direction vector
                distance = np.sqrt(direction_x ** 2 + direction_y ** 2)
                if distance == 0:
                    continue  # Skip points that are already at the center

                norm_direction_x = direction_x / distance
                norm_direction_y = direction_y / distance

                # Move the point inward by the step length
                new_x = x + norm_direction_x * step_length
                new_y = y + norm_direction_y * step_length

                # Stop if moving the point exceeds the center
                if distance < step_length:
                    continue

                new_layer.append((new_x, new_y))

            if not new_layer:
                break  # Stop if no new points can be generated

            layers.append(new_layer)

        return layers
    
    # Slice the object into layers
    #slices = Slicer_backend.slicing_object(object, layer_height=1)

    # Get the points from each slice
    # Sort points clockwise
    #sortedpoints = Slicer_backend.clockwise_sort(points)

    # Find the outer and inner perimeters
    #outer_perimeters, inner_perimeters = Slicer_backend.find_outer_inner_perimeters(sortedpoints)
    
    
    # Generate additional points along the outer perimeter of the first slice
    #sorted_first_slice_gen_o = Slicer_backend.generate_points(outer_perimeters[0], dist_b_points=2)

    #layers = Slicer_backend.generate_inward_steps(sorted_first_slice_gen_o, 0.5)

    # Plot the results for the first slice
    #outer_x, outer_y = zip(*outer_perimeters[0])
    #x_1_slice, y_1_slice = zip(*sorted_first_slice_gen_o)
    #inner_x, inner_y = zip(*inner_perimeters[0])
    #plt.scatter(points[0], points[1])
    #plt.scatter(x_1_slice, y_1_slice)
    #print(x_1_slice)
    #plt.scatter(outer_x, outer_y)
    #startPoint = random_start(outer_perimeters[0])
    #plt.scatter(float(startPoint[1][0]), float(startPoint[1][1]), linewidths=5)
    
    #sortedpoints = Slicer_backend.clockwise_sort(inner_perimeters)
    #outer_perimeters, inner_perimeters = Slicer_backend.find_outer_inner_perimeters(inner_perimeters)
    #outer_x, outer_y = zip(*outer_perimeters[0])
    #plt.scatter(outer_x, outer_y)
    
    
   

    #plt.figure()
    #for layer in layers:
    #    x, y = zip(*layer)
    #    plt.plot(x, y, marker='o')
    '''
