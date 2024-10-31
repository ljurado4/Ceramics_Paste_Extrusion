import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from stl import mesh
from scipy.spatial import ConvexHull
import random

class SlicerBackend:
    
    @staticmethod
    def read_file(filepath):
        # Load the STL file and return the mesh object
        solid_body = mesh.Mesh.from_file(filepath)
        return solid_body
    
    @staticmethod
    def show_file(filepath):
        # Visualize the STL file using pyvista
        file = pv.read(filepath)
        return file.plot()

    @staticmethod
    def get_points_per_slice(solid_body, z_height):
        # Extract boundary points of a slice at a given z-height
        boundary = []
        for triangles in solid_body.vectors:
            for points in triangles:
                # Check if point belongs to the current slice (layer at z_height)
                if points[2] <= z_height:
                    # Only add unique points
                    if [points[0], points[1]] not in boundary:
                        boundary.append([points[0], points[1]]) 
        return boundary

class InfillPatterns:
    
    @staticmethod
    def linear_infill(canvas_size, line_spacing):
        # Generate a zigzag linear infill pattern across the square shape
        
        x_min, y_min = 0, 0
        x_max, y_max = canvas_size[0], canvas_size[1]
        
        lines = []
        y = y_min
        
        while y <= y_max:
            # Move from left to right
            lines.append([(x_min, y), (x_max, y)])
            y += line_spacing
            
            if y > y_max:
                break
            
            # Move from right to left on the next line down
            lines.append([(x_max, y), (x_min, y)])
            y += line_spacing

        return lines

    @staticmethod
    def cross_hatching_infill(canvas_size, line_length, num_lines):
        # Generate cross-hatching pattern for infill

        def generate_random_start_point(canvas_size):
            x = random.uniform(0, canvas_size[0])
            y = random.uniform(0, canvas_size[1])
            return (x, y)

        def generate_line(start_point, angle, length):
            x_start, y_start = start_point
            x_end = x_start + length * np.cos(np.radians(angle))
            y_end = y_start + length * np.sin(np.radians(angle))
            return [(x_start, y_start), (x_end, y_end)]
        
        lines = []
        angle = 0
        for _ in range(num_lines):
            start_point = generate_random_start_point(canvas_size)
            line = generate_line(start_point, angle, line_length)
            lines.append(line)
            angle = 180 if angle == 0 else 0  # Alternate the angle
        return lines

    @staticmethod
    def radial_infill(points, step_length):
        # Generate radial infill by shrinking layers inward step by step

        # Calculate the Convex Hull to get the outermost boundary
        hull = ConvexHull(points)
        boundary_points = [points[i] for i in hull.vertices]

        # Calculate the approximate center point
        center_x = np.mean([x for x, y in boundary_points])
        center_y = np.mean([y for x, y in boundary_points])
        
        # Initialize the radial infill layers
        layers = [boundary_points]
        
        while True:
            new_layer = []
            for x, y in layers[-1]:
                # Calculate the direction vector towards the center
                direction_x = center_x - x
                direction_y = center_y - y
                distance = np.sqrt(direction_x ** 2 + direction_y ** 2)
                
                # Stop if points are close to the center
                if distance < step_length:
                    continue

                # Normalize the direction vector
                norm_x = direction_x / distance
                norm_y = direction_y / distance

                # Move each point inward by step_length
                new_x = x + norm_x * step_length
                new_y = y + norm_y * step_length
                new_layer.append((new_x, new_y))

            # If not enough points for a new boundary, stop
            if len(new_layer) < 3:
                break
            
            # Use Convex Hull to maintain the shape for the new layer
            hull_layer = ConvexHull(new_layer)
            layers.append([new_layer[i] for i in hull_layer.vertices])

        return layers

    @staticmethod
    def adaptive_spiral_infill(points, step_length):
        hull = ConvexHull(points)
        boundary_points = [points[i] for i in hull.vertices]

        center_x = np.mean([x for x, y in boundary_points])
        center_y = np.mean([y for x, y in boundary_points])
        
        spiral_path = []
        radius_decrement = step_length * 0.1
        current_boundary = boundary_points

        while len(current_boundary) > 2:
            new_layer = []
            for x, y in current_boundary:
                direction_x = center_x - x
                direction_y = center_y - y
                distance = np.sqrt(direction_x**2 + direction_y**2)
                
                if distance < radius_decrement:
                    continue
                
                norm_x = direction_x / distance
                norm_y = direction_y / distance
                new_x = x + norm_x * radius_decrement
                new_y = y + norm_y * radius_decrement
                new_layer.append((new_x, new_y))

            if len(new_layer) < 3:
                break
            
            spiral_path.extend(new_layer)
            current_boundary = new_layer

        return spiral_path

def convert_to_gcode(points):
    gcode_commands = ["G21 ; Set units to millimeters", "G90 ; Use absolute positioning", "G1 F1500 ; Set feedrate"]
    
    for point in points:
        if isinstance(point, list):
            for sub_point in point:
                gcode_commands.append(f"G1 X{sub_point[0]:.4f} Y{sub_point[1]:.4f}")
        else:
            gcode_commands.append(f"G1 X{point[0]:.4f} Y{point[1]:.4f}")
    
    return gcode_commands

def main(filepath, infill_type):
    solid_body = SlicerBackend.read_file(filepath)
    SlicerBackend.show_file(filepath)
    
    slice_points = SlicerBackend.get_points_per_slice(solid_body, z_height=0.8)
    
    if infill_type == "linear":
        points = InfillPatterns.linear_infill(canvas_size=(200, 200), line_spacing=0.5)
    elif infill_type == "cross_hatching":
        points = InfillPatterns.cross_hatching_infill(canvas_size=(200, 200), line_length=100, num_lines=10)
    elif infill_type == "radial":
        points = InfillPatterns.radial_infill(slice_points, step_length=0.25)
    elif infill_type == "adaptive_spiral":
        points = InfillPatterns.adaptive_spiral_infill(slice_points, step_length=0.25)
    else:
        raise ValueError("Invalid infill type selected.")
    
    gcode_commands = convert_to_gcode(points)
    
    output_file_path = "/Users/lizbethjurado/Keck/STL/GCODE/generated.gcode"
    
    with open(output_file_path, "w") as gcode_file:
        for command in gcode_commands:
            gcode_file.write(command + "\n")
    
    plt.figure()
    if infill_type in ["cross_hatching", "linear"]:
        for line in points:
            (x_start, y_start), (x_end, y_end) = line
            plt.plot([x_start, x_end], [y_start, y_end], marker='o')
    else:
        if isinstance(points[0], list):
            for layer in points:
                xPoints = [point[0] for point in layer]
                yPoints = [point[1] for point in layer]
                plt.plot(xPoints, yPoints, marker='o')
        else:
            xPoints = [point[0] for point in points]
            yPoints = [point[1] for point in points]
            plt.plot(xPoints, yPoints, marker='o')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    filepath = '/Users/lizbethjurado/Keck/STL/0.5in cube 1.STL'
    infill_type = input("Select infill type (linear, cross_hatching, radial, adaptive_spiral): ")
    main(filepath, infill_type)
