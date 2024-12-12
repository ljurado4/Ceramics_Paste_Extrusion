import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from stl import mesh
from scipy.spatial import ConvexHull
import random
import os

class SlicerBackend:
    @staticmethod
    def read_file(filepath):
        # Load the STL file from the given path and return it as a mesh object
        return mesh.Mesh.from_file(filepath)

    @staticmethod
    def show_file(filepath):
        # Use PyVista to display the 3D model for visualization
        file = pv.read(filepath)
        return file.plot()

    @staticmethod
    def get_points_per_layer(solid_body, z_heights):
        # Slice the 3D model at different Z heights and get the boundary points for each layer
        layers = []
        for z_height in z_heights:
            boundary = []
            for triangles in solid_body.vectors:
                # Check where each triangle intersects with the current Z layer
                intersections = []
                for i in range(3):
                    p1, p2 = triangles[i], triangles[(i + 1) % 3]
                    # If the edge crosses the Z height, calculate the intersection
                    if (p1[2] - z_height) * (p2[2] - z_height) <= 0:
                        t = (z_height - p1[2]) / (p2[2] - p1[2])
                        intersect_point = p1 + t * (p2 - p1)
                        intersections.append(intersect_point[:2])  # We only care about X and Y
                # Add unique points to the boundary
                if len(intersections) == 2:
                    for point in intersections:
                        if tuple(point) not in boundary:
                            boundary.append(tuple(point))
            layers.append(boundary)
        return layers

class InfillPatterns:
    @staticmethod
    def linear_infill(canvas_size, line_spacing=1.0):
        # Create a simple zigzag pattern of lines for infill
        x_min, y_min = 0, 0
        x_max, y_max = canvas_size[0], canvas_size[1]
        path = []
        y = y_min
        direction = 1
        while y <= y_max:
            # Alternate between left-to-right and right-to-left
            if direction == 1:
                path.append((x_min, y))
                path.append((x_max, y))
            else:
                path.append((x_max, y))
                path.append((x_min, y))
            y += line_spacing
            direction *= -1
        return path

    @staticmethod
    def cross_hatching_infill(canvas_size, line_length=100, num_lines=10):
        # Create a pattern with crisscrossing lines
        def generate_random_start_point(canvas_size):
            # Start points are randomly picked within half the canvas size
            x = random.uniform(0, canvas_size[0] / 2)
            y = random.uniform(0, canvas_size[1] / 2)
            return (x, y)

        def generate_line(start_point, angle, length):
            # Create a line at a given angle and length starting from the point
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
            # Alternate between horizontal and vertical angles
            angle = 180 if angle == 0 else 0
        return lines

    @staticmethod
    def radial_infill_square(points, p_width=1.0):
        # Create a pattern of shrinking squares that look like concentric layers
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
            # Shrink the square inward
            x_min += p_width
            x_max -= p_width
            y_min += p_width
            y_max -= p_width
        return layers

    @staticmethod
    def adaptive_spiral_infill(points, step_length=0.5):
        # Create a spiral pattern that shrinks inward
        points = [p for p in points if len(p) == 2]  # Ensure we only have [x, y] pairs
        if len(points) >= 3:
            hull = ConvexHull(points)
            boundary_points = [points[i] for i in hull.vertices]
        else:
            return []

        # Start at the center and spiral inward
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
    # Convert the path of points into G-code commands for the printer
    gcode_commands = [
        "N0 G21 ; Set units to millimeters",
        "N10 G90 ; Use absolute positioning",
        f"N20 G1 F{speed} ; Set feedrate to {speed} mm/min"
    ]

    def bead_area_calculation(H, W):
        # Calculate the bead area for accurate extrusion if needed
        return H * W - (H**2 * (1 - (np.pi / 4)))

    n_value = 30
    extrusion_amount = 0
    H = 0.4
    W = 0.84
    Dn = 0.84
    extrusion_param = "P" if use_ceramic_extrusion else "E"  # Use P for ceramic extrusion
    for i in range(len(path) - 1):
        x_start, y_start = path[i]
        x_end, y_end = path[i + 1]
        distance = np.linalg.norm([x_end - x_start, y_end - y_start])
        if bead_area_formula:
            A_bead = bead_area_calculation(H, W)
            extrusion_amount += A_bead * distance * Dn
        else:
            extrusion_amount += distance * extrusion_rate
        gcode_commands.append(
            f"N{n_value} G1 X{x_end:.4f} Y{y_end:.4f} {extrusion_param}{extrusion_amount:.4f} ; Move to point"
        )
        n_value += 10
    return gcode_commands

def main(filepath, infill_type, speed=60, extrusion_rate=0.5, bead_area_formula=False, use_ceramic_extrusion=False):
    # Read the STL file and display the model
    solid_body = SlicerBackend.read_file(filepath)
    SlicerBackend.show_file(filepath)

    # Define slicing parameters
    layer_height = 0.2
    z_max = max(solid_body.vectors[:, :, 2].flatten())
    z_heights = np.arange(0, z_max, layer_height)

    # Slice the model and process each layer
    layers = SlicerBackend.get_points_per_layer(solid_body, z_heights)
    gcode_commands = []
    for i, slice_points in enumerate(layers):
        print(f"Processing layer {i + 1} at Z={z_heights[i]:.2f}")

        # Select the infill pattern based on user input
        if infill_type == "1" or infill_type.lower() == "a":
            points = InfillPatterns.linear_infill((20, 20), 1.0)
        elif infill_type == "2" or infill_type.lower() == "b":
            points = InfillPatterns.cross_hatching_infill((20, 20), 10, 10)
        elif infill_type == "3" or infill_type.lower() == "c":
            points = InfillPatterns.radial_infill_square(slice_points, 1.0)
        elif infill_type == "4" or infill_type.lower() == "d":
            points = InfillPatterns.adaptive_spiral_infill(slice_points, 0.5)
        else:
            raise ValueError("Invalid infill type selected.")

        # Generate G-code for the current layer and add it to the overall commands
        layer_gcode = convert_to_gcode(points, speed, extrusion_rate, bead_area_formula, use_ceramic_extrusion)
        gcode_commands.extend(layer_gcode)

    # Save the G-code for both Lizbeth and Zack
    lizbeth_output_path = '/Users/lizbethjurado/Keck/Slicing Cube/GCode_Output/generated.gcode'
    zack_output_path = 'C:\\Users\\zach\\Desktop\\SlicingCube\\GCode_Output\\generated.gcode'
    with open(lizbeth_output_path, 'w') as file:
        file.write("\n".join(gcode_commands))
    print(f"G-code has been saved to Lizbeth's path: {lizbeth_output_path}")

    with open(zack_output_path, 'w') as file:
        file.write("\n".join(gcode_commands))
    print(f"G-code has been saved to Zack's path: {zack_output_path}")

if __name__ == "__main__":
    # Define the STL file paths for Lizbeth and Zack
    lizbeth_filepath = '/Users/lizbethjurado/Keck/Slicing Cube/TwentyMMcube.stl'
    zack_filepath = 'C:\\Users\\zach\\Desktop\\SlicingCube\\TwentyMMcube.stl'

    # Choose which file path to use (defaulting to Lizbeth's for now)
    filepath = lizbeth_filepath  # Change to zack_filepath if running on Zack's system

    # Run the slicer with the chosen file and parameters
    main(filepath, infill_type="1", speed=60, extrusion_rate=0.5, bead_area_formula=True, use_ceramic_extrusion=True)