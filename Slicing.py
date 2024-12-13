import matplotlib.pyplot as plt
import pyvista as pv
from stl import mesh
import numpy as np
from shapely.geometry import Polygon
import math
#from scipy.spatial import ConvexHull
#import random

class Slicer_backend:

    @staticmethod
    def read_file_show(filepath):
        solid_body = mesh.Mesh.from_file(filepath)
        file = pv.read(filepath)
        return solid_body, file.plot()

    @staticmethod
    def get_points_per_slice(solid_body, maxLayerHeight, minLayerHeight = 0):
        points = solid_body.vectors.reshape(-1, 3)  # Reshape for easier access
        z_condition = (minLayerHeight <= points[:, 2]) & (points[:, 2] <= maxLayerHeight) # Create a boolean array for z conditions
        valid_points = points[z_condition]  # Filter points based on z

        # Create a set of (x, y) tuples
        boundary = np.array(list(set(map(tuple, valid_points[:, :2]))))  # Only keep x and y

        # Calculate the centroid of the points
        centroid = np.mean(boundary, axis=0)

        # Function to calculate angle from centroids
        def angle_from_centroid(point):
            return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
        
        # Sort points by angle relative to the centroid
        sorted_boundary = boundary[np.argsort([angle_from_centroid(p) for p in boundary])]

        return sorted_boundary.tolist()  # Convert back to list

    @staticmethod
    def scale_shape_down(perimeter_points, offset_distance=0.8):
        # Create a polygon from the perimeter points
        print(perimeter_points)
        polygon = Polygon(perimeter_points)

        # Offset the polygon inward
        scaled_polygon = polygon.buffer(-offset_distance)

        # Check if the resulting polygon is valid
        if scaled_polygon.is_empty:
            return []  # Return empty if the polygon is invalid

        # Extract the new perimeter points from the scaled polygon
        new_perimeter = np.array(scaled_polygon.exterior.coords)

        return new_perimeter.tolist()  # Convert back to list
        
    @staticmethod
    def fill_in_lines(perimeter_points, spacing=0.8):
        filled_points = set()  # Use a set to avoid duplicates

        # Normalize the direction parameter
        for i in range(len(perimeter_points)):
            start_point = np.array(perimeter_points[i])
            end_point = np.array(perimeter_points[(i + 1) % len(perimeter_points)])
            
            # Calculate the distance between the two points
            distance = np.linalg.norm(end_point - start_point)
        
            # Generate points only in the specified direction
            num_points = int(np.ceil(distance / spacing))
            for j in range(num_points):
                interpolated_point = start_point + (end_point - start_point) * (j / num_points)
                point = (round(interpolated_point[0], 6), round(start_point[1], 6))  # Maintain y-coordinate
                filled_points.add(point)
                point = (round(start_point[0], 6), round(interpolated_point[1], 6))  # Maintain x-coordinate
                filled_points.add(point)
                    
        # Convert the set to a list and sort it
        sorted_points = sorted(filled_points, key=lambda x: (x[1], x[0] if int(x[1] * 100) % 2 == 0 else -x[0]))

        return np.array(sorted_points).tolist()  # Convert back to list

def calculate_centroid(points): 
    # Calculate the centroid of the points
    x_sum = sum(p[0] for p in points)
    y_sum = sum(p[1] for p in points)
    return (x_sum / len(points), y_sum / len(points))

def sort_points_counter_clockwise(points, start_point):
    centroid = calculate_centroid(points)
    
    def angle_from_centroid(point):
        # Using math.atan2 to compute the angle relative to the centroid
        dx = point[0] - centroid[0]
        dy = point[1] - centroid[1]
        return math.atan2(dy, dx)
    
    sorted_points = sorted(points, key=angle_from_centroid)
    start_index = sorted_points.index(start_point)
    # Rotate the list so that the closest point is the starting point
    return sorted_points[start_index:] + sorted_points[:start_index]



def generate_lines(infill_points, direction='horizontal'):
    lines = []
    if direction == 'horizontal':
        j = -1 * len(infill_points) // 4
        for i in range(len(infill_points) // 2):
            if i % 2 == 0:
                lines.append((infill_points[i], infill_points[j]))
            else:
                lines.append((infill_points[j], infill_points[i]))
            j-= 1
    elif direction == 'vertical':
        j = len(infill_points) // 4
        for i in range(0, -1 * len(infill_points) // 2, -1):
            if i % 2 == 0:
                lines.append((infill_points[i], infill_points[j]))
            else:
                lines.append((infill_points[j], infill_points[i]))
            j += 1
    elif direction == 'angled':
        for i in range(len(infill_points) // 2 + 1):
            if i == 0 or i == len(infill_points) // 2:
                lines.append((infill_points[i], infill_points[i]))
            if i % 2 == 0:
                lines.append((infill_points[-i], infill_points[i]))
            else:
                lines.append((infill_points[i], infill_points[-i]))
    return lines

def random_start(perimeter, prevPoint = None):
    pointList = [[i, j] for i, j in perimeter]
    if prevPoint == None:
        return [1, pointList[0]]
    index = prevPoint
    if index == 0:
        index += len(pointList) // 2
    else:
        index = 0
    return [index, pointList[index]]

def find_closest_point_index(points, key_point):
    closest_index = -1
    min_distance = float('inf')

    for i, point in enumerate(points):
        # Calculate Euclidean distance
        distance = math.sqrt((point[0] - key_point[0]) ** 2 + (point[1] - key_point[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_index = i

    return closest_index

def convert_to_gcode(points):
    #Dn = 0.84   mm diameter of nozzle
    #U = 10   mm/s print speed
    H = 0.4  # mm standoff distance
    W = 0.84  # desired width
    Ext_mult = 1
    Pvalue = 0
    Nval = 0
    g_commands = []
    
    A_bead = H * W - (H**2 * (1 - (np.pi / 4)))
    
    for i in range(len(points)):
        point = points[i]
        print(point)
        if len(point) == 3:
            (x_start, y_start, z_start) = point
            g_commands.append(f"N{Nval} G0 X{x_start:.4f} Y{y_start:.4f} Z{z_start:.4f}")
            Nval += 10
        else:
            (x_start, y_start) = point
        if len(g_commands) == 0:
            g_commands = ["G21 ; Set units to millimeters", "G90 ; Use absolute positioning", "G1 F1500 ; Set feedrate"]
            g_commands.append(f"N{Nval} G0 X{x_start:.4f} Y{y_start:.4f}")
            Nval += 10
            continue
        if f"G1 X{x_start:.4f} Y{y_start:.4f}" not in g_commands[-1] and f"G0 X{x_start:.4f} Y{y_start:.4f}" not in g_commands[-1]:
            start_point = np.array(points[i-1])
            end_point = np.array(point)
            E_distance = np.linalg.norm(end_point - start_point)
            if len(points) > 1:
                Pvalue = A_bead * E_distance * Ext_mult + Pvalue
                #if E_distance < W:
                    #g_commands.append(f"N{Nval} G0 X{x_start:.4f} Y{y_start:.4f}")
                    #Nval += 10
                    #continue
                g_commands.append(f"N{Nval} G1 X{x_start:.4f} Y{y_start:.4f} P{Pvalue}")
                Nval += 10

    return g_commands

if __name__ == "__main__":
    current_layer_height = 0
    bead_height = .4
    max_layer_height = bead_height
    currentStartPoint = None
    nextStartPoint = None
    points = []
    g_code_path = []
    directionDic = {"1":'vertical', "2":'horizontal', "3":'angled'}
    direction = directionDic[input("Choose infill method:\n1. Vertical\n2. Horizontal\n3. Angled\n")]
    
    # Load the 3D object and visualize it
    object, visualization = Slicer_backend.read_file_show(filepath=r"C:\Users\zzcro\Desktop\Lab_Assignments\Keck\Ceramics_Paste_Extrusion\TwentyMMcube.stl")
    ''' Slice Per Layer '''
    vectors = object.vectors.reshape(-1,3)
    maxZIndex = np.argmax(vectors[:,2])
    maxZ = vectors[maxZIndex][2]
    
    while(current_layer_height < maxZ):
        points = points if (temp_points := Slicer_backend.get_points_per_slice(object, max_layer_height, current_layer_height)) == [] else temp_points
        GCodepoints = points
        plt.figure()
        
        infill_points = Slicer_backend.fill_in_lines(Slicer_backend.scale_shape_down(points))
        if currentStartPoint is None:
            currentStartPoint = random_start(points)
        else:
            currentStartPoint = nextStartPoint
            currentStartPoint[1].append(current_layer_height)
        GCodepoints = sort_points_counter_clockwise(GCodepoints, currentStartPoint[1][:2])
        GCodepoints.append(GCodepoints[0])
        
        infillStartIndex = find_closest_point_index(infill_points, GCodepoints[0])
        infill_points = sort_points_counter_clockwise(infill_points, infill_points[infillStartIndex])
        
        plt.scatter(currentStartPoint[1][0], currentStartPoint[1][1], c='red')
        
    
        GCodepoints.append(infill_points[0]) 
        lines = generate_lines(infill_points, direction)
        for i in range(len(lines)):
            line = lines[i]
            [(start_x, start_y), (end_x, end_y)] = line
            if line[0] == line[1]:
                GCodepoints.extend([[start_x, start_y]])
            else:
                GCodepoints.extend([[start_x, start_y]])
                GCodepoints.extend([[end_x, end_y]]) 
    
        x_points = [point[0] for point in GCodepoints]
        y_points = [point[1] for point in GCodepoints]
        plt.plot(x_points, y_points)
        
        current_layer_height += bead_height
        max_layer_height += bead_height
        
        if current_layer_height < maxZ:
            nextStartPoint = random_start(points if (temp_points := Slicer_backend.get_points_per_slice(object, max_layer_height, current_layer_height)) == [] else temp_points, currentStartPoint[0])
            g_code_path.extend(GCodepoints)
            g_code_path.append(nextStartPoint[1])
        else:
            g_code_path.extend(GCodepoints)
        
        
        output_file_path = r"C:\Users\zzcro\Desktop\Lab_Assignments\Keck\Ceramics_Paste_Extrusion\generated.gcode"
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()
    g_commands = convert_to_gcode(g_code_path)
        

    with open(output_file_path, "w") as gcode_file:
        for command in g_commands:
            gcode_file.write(command + "\n")
    
    
    
    
    
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