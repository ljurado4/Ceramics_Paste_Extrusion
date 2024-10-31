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
    def linear_infill(points):
        # No special pattern, just return the points as they are for linear infill
        return points

    @staticmethod
    def cross_hatching_infill(canvas_size, line_length, num_lines):
        # Generate cross-hatching pattern for infill

        # Function to get random starting points on the canvas
        def generate_random_start_point(canvas_size):
            x = random.uniform(0, canvas_size[0])
            y = random.uniform(0, canvas_size[1])
            return (x, y)

        # Function to generate a line given a start point, angle, and length
        def generate_line(start_point, angle, length):
            x_start, y_start = start_point
            x_end = x_start + length * np.cos(np.radians(angle))
            y_end = y_start + length * np.sin(np.radians(angle))
            return [(x_start, y_start), (x_end, y_end)]
        
        # Alternate between two angles (0° and 180°) to create the cross-hatching effect
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

        # Calculate the approximate center point of the shape
        center_x = sum(x for x, y in points) / len(points)
        center_y = sum(y for x, y in points) / len(points)
        center = (center_x, center_y)
        
        # Initialize the outer boundary using Convex Hull to handle various shapes
        hull = ConvexHull(points)
        boundary_points = [points[i] for i in hull.vertices]
        
        # Initialize with the outer boundary
        layers = [boundary_points]
        
        while True:
            new_layer = []
            for x, y in layers[-1]:
                # Calculate the direction vector to the center
                direction_x = center_x - x
                direction_y = center_y - y
                distance = np.sqrt(direction_x ** 2 + direction_y ** 2)
                
                if distance < step_length:  # Stop when points reach near the center
                    continue

                # Normalize the direction vector
                norm_direction_x = direction_x / distance
                norm_direction_y = direction_y / distance

                # Move each point inward by step_length
                new_x = x + norm_direction_x * step_length
                new_y = y + norm_direction_y * step_length
                new_layer.append((new_x, new_y))

            if len(new_layer) < 3:  # Stop if not enough points to form a closed loop
                break

            # Use Convex Hull on the new layer to maintain the boundary shape
            hull_layer = ConvexHull(new_layer)
            layers.append([new_layer[i] for i in hull_layer.vertices])

        return layers

def convert_to_gcode(points):
    # Convert the list of points into G-code commands
    gcode_commands = ["G21 ; Set units to millimeters", "G90 ; Use absolute positioning", "G1 F1500 ; Set feedrate"]
    
    # Add the G1 commands to move the printer to each point
    for point in points:
        if isinstance(point, list):
            for sub_point in point:
                gcode_commands.append(f"G1 X{sub_point[0]:.4f} Y{sub_point[1]:.4f}")
        else:
            gcode_commands.append(f"G1 X{point[0]:.4f} Y{point[1]:.4f}")
    
    return gcode_commands

def main(filepath, infill_type):
    # Load and visualize the STL file
    solid_body = SlicerBackend.read_file(filepath)
    SlicerBackend.show_file(filepath)
    
    # Slice the object at a specific layer height (e.g., 0.8)
    slice_points = SlicerBackend.get_points_per_slice(solid_body, z_height=0.8)
    
    # Choose the infill pattern based on user input
    if infill_type == "linear":
        points = InfillPatterns.linear_infill(slice_points)
    elif infill_type == "cross_hatching":
        # Cross-hatching with predefined canvas size and line length
        points = InfillPatterns.cross_hatching_infill(canvas_size=(200, 200), line_length=100, num_lines=10)
    elif infill_type == "radial":
        # Radial infill with step length
        points = InfillPatterns.radial_infill(slice_points, step_length=0.5)
    else:
        raise ValueError("Invalid infill type selected.")
    
    # Convert the points generated by the selected infill method to G-code
    gcode_commands = convert_to_gcode(points)
    
    # Write the generated G-code commands to a file
    output_file_path = "/Users/lizbethjurado/Keck/STL/GCODE/generated.gcode"
    
    with open(output_file_path, "w") as gcode_file:
        for command in gcode_commands:
            gcode_file.write(command + "\n")
    
    # Plot the points to visualize the infill pattern
    plt.figure()
    if infill_type == "cross_hatching":
        # Special case for cross-hatching since it consists of lines
        for line in points:
            (x_start, y_start), (x_end, y_end) = line
            plt.plot([x_start, x_end], [y_start, y_end], marker='o')
    else:
        # Default case for scatter plot of points
        for layer in points:
            xPoints = [point[0] for point in layer]
            yPoints = [point[1] for point in layer]
            plt.scatter(xPoints, yPoints, c='blue')
            plt.plot(xPoints, yPoints)
    
    # Ensure equal aspect ratio and display the grid
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Set the file path and ask the user to select the infill type
    filepath = '/Users/lizbethjurado/Keck/STL/0.5in cube 1.STL'
    infill_type = input("Select infill type (linear, cross_hatching, radial): ")
    main(filepath, infill_type)