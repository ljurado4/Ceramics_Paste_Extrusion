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
        # Generate a continuous snake-like zigzag linear infill pattern across the square shape
        x_min, y_min = 0, 0
        x_max, y_max = canvas_size[0], canvas_size[1]
        
        path = []
        y = y_min
        direction = 1  # Start moving left to right
        
        while y <= y_max:
            if direction == 1:  # Move from left to right
                path.append((x_min, y))
                path.append((x_max, y))
            else:  # Move from right to left
                path.append((x_max, y))
                path.append((x_min, y))
            
            # Move to the next line down
            y += line_spacing
            direction *= -1  # Reverse direction for the next row
        
        return path

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
    def radial_infill_square(points, p_width=0.8):
        # Assuming the points define a square or rectangular outer boundary.
        
        # Calculating the bounding box (min and max x and y) of the points to get initial square dimensions.
        x_coords, y_coords = zip(*points)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # List to store each inward square layer.
        layers = []
        
        # Generating inward squares until they collapse to the center.
        while (x_max - x_min > p_width * 2) and (y_max - y_min > p_width * 2):
            # Define the current layer as a square with corners based on min/max x and y.
            layer = [
                (x_min, y_min),  # Bottom-left corner
                (x_max, y_min),  # Bottom-right corner
                (x_max, y_max),  # Top-right corner
                (x_min, y_max),  # Top-left corner
                (x_min, y_min)   # Close the square by returning to bottom-left
            ]
            layers.append(layer)
            
            # Move inward by the print width (p_width).
            x_min += p_width
            x_max -= p_width
            y_min += p_width
            y_max -= p_width
        
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

def convert_to_gcode(path):
    # Convert the continuous path of points into G-code commands
    gcode_commands = ["G21 ; Set units to millimeters", "G90 ; Use absolute positioning", "G1 F1500 ; Set feedrate"]
    
    # Add the G1 commands to move the printer to each point in the continuous path
    for x, y in path:
        gcode_commands.append(f"G1 X{x:.4f} Y{y:.4f}")
    
    return gcode_commands

def main(filepath, infill_type):
    # Load and visualize the STL file
    solid_body = SlicerBackend.read_file(filepath)
    SlicerBackend.show_file(filepath)
    
    # Slice the object at a specific layer height (e.g., 0.8)
    slice_points = SlicerBackend.get_points_per_slice(solid_body, z_height=0.8)
    
    # Choose the infill pattern based on user input
    if infill_type == "1" or infill_type.lower() == "a":
        points = InfillPatterns.linear_infill(canvas_size=(200, 200), line_spacing=0.5)
    elif infill_type == "2" or infill_type.lower() == "b":
        points = InfillPatterns.cross_hatching_infill(canvas_size=(200, 200), line_length=100, num_lines=10)
    elif infill_type == "3" or infill_type.lower() == "c":
        points = InfillPatterns.radial_infill_square(slice_points, p_width=0.8)
    elif infill_type == "4" or infill_type.lower() == "d":
        points = InfillPatterns.adaptive_spiral_infill(slice_points, step_length=0.25)
    else:
        raise ValueError("Invalid infill type selected.")
    
    # Convert the points generated by the selected infill method to G-code
    gcode_commands = convert_to_gcode(points)
    
    # Write the generated G-code commands to a file
    output_file_path = "/Users/lizbethjurado/Keck/STL/GCODE/generated.gcode"
    
    with open(output_file_path, "w") as gcode_file:
        for command in gcode_commands:
            gcode_file.write(command + "\n")
    
    # Plot the continuous path for linear infill
    plt.figure()
    if infill_type in ["1", "a", "A"]:
        xPoints, yPoints = zip(*points)
        plt.plot(xPoints, yPoints, marker='o')
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
    print("Select infill type:")
    print("1 or A - Linear Infill")
    print("2 or B - Cross Hatching Infill")
    print("3 or C - Radial Infill")
    print("4 or D - Adaptive Spiral Infill")
    infill_type = input("Enter your choice: ")
    main(filepath, infill_type)
