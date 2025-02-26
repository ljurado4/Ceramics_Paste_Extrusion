import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from stl import mesh
from scipy.spatial import ConvexHull
import os

class SlicerBackend:
    @staticmethod
    def read_file(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"STL file not found: {filepath}")
        return mesh.Mesh.from_file(filepath)

    @staticmethod
    def show_file(filepath):
        file = pv.read(filepath)
        return file.plot()

    @staticmethod
    def get_points_per_layer(solid_body, z_heights):
        layers = []
        for z_height in z_heights:
            boundary = []
            for triangles in solid_body.vectors:
                intersections = []
                for i in range(3):
                    p1, p2 = triangles[i], triangles[(i + 1) % 3]

                    # Ensure we don't divide by zero
                    if (p2[2] - p1[2]) == 0:
                        continue

                    if (p1[2] - z_height) * (p2[2] - z_height) <= 0:
                        t = (z_height - p1[2]) / (p2[2] - p1[2])
                        intersect_point = p1 + t * (p2 - p1)
                        intersections.append(intersect_point[:2])

                if len(intersections) == 2:
                    for point in intersections:
                        if tuple(point) not in boundary:
                            boundary.append(tuple(point))

            # Ensure boundary points form a valid shape
            if len(boundary) >= 3:
                hull = ConvexHull(boundary)
                boundary = [boundary[i] for i in hull.vertices]

            layers.append(boundary)
        return layers

def convert_to_gcode(path, speed=20, extrusion_rate=0.5, bead_area_formula=False, is_first_layer=False, previous_extrusion=0):
    gcode_commands = []

    if is_first_layer:
        gcode_commands.append("N0 G1 F20 E900 E-900 ; Initial setup for extrusion")

    def bead_area_calculation(H, W):
        return H * W - (H**2 * (1 - (np.pi / 4)))

    n_value = 10
    extrusion_amount = previous_extrusion
    H = 0.4
    W = 0.84

    for i in range(len(path) - 1):
        x_start, y_start = path[i]
        x_end, y_end = path[i + 1]
        distance = np.linalg.norm([x_end - x_start, y_end - y_start])

        if bead_area_formula:
            A_bead = bead_area_calculation(H, W)
            extrusion_amount += A_bead * distance
        else:
            extrusion_amount += distance * extrusion_rate

        gcode_commands.append(
            f"N{n_value} G1 X{x_end:.4f} Y{y_end:.4f} E{extrusion_amount:.4f} ; Move with extrusion"
        )
        n_value += 10

    return gcode_commands, extrusion_amount

def main(filepath, infill_type, speed=20, extrusion_rate=0.5, bead_area_formula=False):
    # Ensure STL file exists
    solid_body = SlicerBackend.read_file(filepath)
    SlicerBackend.show_file(filepath)

    layer_height = 0.2
    z_max = max(solid_body.vectors[:, :, 2].flatten())
    z_heights = np.arange(0, z_max, layer_height)

    print(f"Z-Max: {z_max}, Layer Height: {layer_height}")
    print(f"Z-Heights: {z_heights}")

    layers = SlicerBackend.get_points_per_layer(solid_body, z_heights)
    gcode_commands = []
    extrusion_amount = 0

    for i, slice_points in enumerate(layers):
        if not slice_points:
            continue

        gcode_commands.append(f"N{10 * i + 100} G1 Z{z_heights[i]:.2f} ; Move to Z height")

        layer_gcode, extrusion_amount = convert_to_gcode(
            slice_points, speed, extrusion_rate, bead_area_formula, is_first_layer=(i == 0), previous_extrusion=extrusion_amount
        )
        
        gcode_commands.extend(layer_gcode)

    lizbeth_output_path = '/Users/lizbethjurado/Keck/Slicing Cube/GCode_Output/generated.gcode'
    zack_output_path = 'C:\\Users\\zach\\Desktop\\SlicingCube\\GCode_Output\\generated.gcode'

    # Ensure output directory exists before writing
    if lizbeth_output_path and os.path.dirname(lizbeth_output_path):
        os.makedirs(os.path.dirname(lizbeth_output_path), exist_ok=True)
    
    if zack_output_path and os.path.dirname(zack_output_path):
        os.makedirs(os.path.dirname(zack_output_path), exist_ok=True)

    # Write G-code to Lizbeth's path
    if lizbeth_output_path:
        with open(lizbeth_output_path, 'w') as file:
            file.write("\n".join(gcode_commands))
        print(f"G-code saved to: {lizbeth_output_path}")

    # Write G-code to Zach's path (if on Windows)
    if zack_output_path and os.name == 'nt':  # Ensures it runs only on Windows
        with open(zack_output_path, 'w') as file:
            file.write("\n".join(gcode_commands))
        print(f"G-code saved to: {zack_output_path}")

if __name__ == "__main__":
    lizbeth_filepath = '/Users/lizbethjurado/Keck/Slicing Cube/STLs/I Letter Starwars v2.stl'
    ## zack_filepath = 'C:\\Users\\zach\\Desktop\\SlicingCube\\TwentyMMcube.stl'
    filepath = lizbeth_filepath
    main(filepath, infill_type="1", speed=20, extrusion_rate=0.5, bead_area_formula=True)