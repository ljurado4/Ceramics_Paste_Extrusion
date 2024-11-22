import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from stl import mesh
from scipy.spatial import ConvexHull
import random
import os

class SlicerBackend:
    # This class handles reading, visualizing, and processing STL files.
    
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
        # Extracting the boundary points of a slice at a given z-height
        boundary = []
        for triangles in solid_body.vectors:
            for points in triangles:
                # Check if point belongs to the current slice (layer at z_height)
                if points[2] <= z_height:
                    # Only add unique [x, y] points
                    xy_point = [points[0], points[1]]
                    if xy_point not in boundary:
                        boundary.append(xy_point)
        return boundary

class InfillPatterns:
    # This class generates different infill patterns for 3D printing.

    @staticmethod
    def linear_infill(canvas_size, line_spacing=1.0):
        # Generate a zigzag linear infill pattern with smaller spacing
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
            
            # Move to the next line with smaller spacing
            y += line_spacing  # Reduced line_spacing
            direction *= -1  # Alternate direction for zigzag
        
        return path

    @staticmethod
    def cross_hatching_infill(canvas_size, line_length=100, num_lines=10):
        # Generate a cross-hatching infill with more controlled density

        def generate_random_start_point(canvas_size):
            x = random.uniform(0, canvas_size[0] / 2)  # Limit to smaller range
            y = random.uniform(0, canvas_size[1] / 2)
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
            angle = 180 if angle == 0 else 0  # Alternate angle
        return lines

    @staticmethod
    def radial_infill_square(points, p_width=1.0):
        # Reduced p_width for a denser radial infill
        
        x_coords, y_coords = zip(*points)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        layers = []
        while (x_max - x_min > p_width * 2) and (y_max - y_min > p_width * 2):
            layer = [
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
                (x_min, y_min)
            ]
            layers.append(layer)
            x_min += p_width
            x_max -= p_width
            y_min += p_width
            y_max -= p_width
        
        return layers

    @staticmethod
    def adaptive_spiral_infill(points, step_length=0.5):
        # Use smaller step_length for tighter spiral infill
        points = [p for p in points if len(p) == 2]  # Ensure [x, y] pairs only

        if len(points) >= 3:
            hull = ConvexHull(points)
            boundary_points = [points[i] for i in hull.vertices]
        else:
            return []

        center_x = np.mean([x for x, y in boundary_points])
        center_y = np.mean([y for x, y in boundary_points])
        
        spiral_path = []
        radius_decrement = step_length * 0.5
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

def convert_to_gcode(path, speed=60, extrusion_rate=0.5, bead_area_formula=False, use_ceramic_extrusion=False):
    """
    Generate G-code from a given path.
    Parameters:
        path: List of (x, y) points defining the movement path.
        speed: Movement speed in mm/min.
        extrusion_rate: Extrusion rate (scaling factor for E or P values).
        bead_area_formula: Boolean to enable advanced bead area calculation.
        use_ceramic_extrusion: Boolean to switch between "E" (regular extrusion) and "P" (ceramic extrusion).
    """
    gcode_commands = [
        "N0 G21 ; Set units to millimeters",
        "N10 G90 ; Use absolute positioning",
        f"N20 G1 F{speed} ; Set feedrate to {speed} mm/min"
    ]
    
    def bead_area_calculation(H, W):
        return H * W - (H**2 * (1 - (np.pi / 4)))
    
    n_value = 30  # Start with N value 30
    extrusion_amount = 0
    H = 0.4  # Layer height
    W = 0.84  # Bead width
    Dn = 0.84  # Density scaling factor
    
    extrusion_param = "P" if use_ceramic_extrusion else "E"
    
    for i in range(len(path) - 1):
        x_start, y_start = path[i]
        x_end, y_end = path[i + 1]
        
        # Calculate distance between two points
        distance = np.linalg.norm([x_end - x_start, y_end - y_start])
        
        if bead_area_formula:
            A_bead = bead_area_calculation(H, W)
            extrusion_amount += A_bead * distance * Dn
        else:
            extrusion_amount += distance * extrusion_rate
        
        gcode_commands.append(
            f"N{n_value} G1 X{x_end:.4f} Y{y_end:.4f} {extrusion_param}{extrusion_amount:.4f} ; Move to point"
        )
        n_value += 10  # Increment N value
    
    return gcode_commands

def main(filepath, infill_type, speed=60, extrusion_rate=0.5, bead_area_formula=False, use_ceramic_extrusion=False):
    solid_body = SlicerBackend.read_file(filepath)
    SlicerBackend.show_file(filepath)
    
    slice_points = SlicerBackend.get_points_per_slice(solid_body, z_height=0.8)
    
    if infill_type == "1" or infill_type.lower() == "a":
        points = InfillPatterns.linear_infill(canvas_size=(20, 20), line_spacing=1.0)
    elif infill_type == "2" or infill_type.lower() == "b":
        points = InfillPatterns.cross_hatching_infill(canvas_size=(20, 20), line_length=10, num_lines=10)
    elif infill_type == "3" or infill_type.lower() == "c":
        points = InfillPatterns.radial_infill_square(slice_points, p_width=1.0)
    elif infill_type == "4" or infill_type.lower() == "d":
        points = InfillPatterns.adaptive_spiral_infill(slice_points, step_length=0.5)
    else:
        raise ValueError("Invalid infill type selected.")
    
    gcode_commands = convert_to_gcode(points, speed, extrusion_rate, bead_area_formula, use_ceramic_extrusion)
    
    # Lizbeth's output file path 
    output_file_path = '/Users/lizbethjurado/Keck/Slicing Cube/GCode_Output/generated.gcode'
    # Zach's output file path 
    # output_file_path = 'C:\\Users\\zach\\Desktop\\SlicingCube\\GCode_Output\\generated.gcode'
    
    with open(output_file_path, 'w') as file:
        file.write("\n".join(gcode_commands))
    print(f"G-code has been saved to: {output_file_path}")

if __name__ == "__main__":
    # Lizbeth's STL input file path 
    filepath = '/Users/lizbethjurado/Keck/Slicing Cube/TwentyMMcube.stl'
    # Zach's STL input file path 
    # filepath = 'C:\\Users\\zach\\Desktop\\SlicingCube\\TwentyMMcube.stl'
    
    main(filepath, infill_type="1", speed=60, extrusion_rate=0.5, bead_area_formula=True, use_ceramic_extrusion=True)