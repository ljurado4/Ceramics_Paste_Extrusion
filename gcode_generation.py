import numpy as np
import random
import matplotlib.pyplot as plt

# Parameters for line generation
canvas_size = (200, 200)  # Define the size of the canvas
line_length = 100  # Length of the lines
number_of_lines = 10  # Number of lines to generate

# Function to generate random starting points within the canvas boundaries
def generate_random_start_point(canvas_size):
    x = random.uniform(0, canvas_size[0])
    y = random.uniform(0, canvas_size[1])
    return (x, y)

# Function to generate a line based on start point and angle
def generate_line(start_point, angle, length):
    x_start, y_start = start_point
    x_end = x_start + length * np.cos(np.radians(angle))
    y_end = y_start + length * np.sin(np.radians(angle))
    return [(x_start, y_start), (x_end, y_end)]

# Function to alternate cross-hatching angles
def generate_cross_hatching_lines(number_of_lines, length, canvas_size):
    lines = []
    angle = 0  # Starting angle (0 or 180 degrees)
    for i in range(number_of_lines):
        start_point = generate_random_start_point(canvas_size)
        line = generate_line(start_point, angle, length)
        lines.append(line)
        # Alternate the angle between 0 and 180 degrees
        angle = 180 if angle == 0 else 0
    return lines

# Generate the lines with cross-hatching pattern
lines = generate_cross_hatching_lines(number_of_lines, line_length, canvas_size)

# Plotting the lines for visualization
plt.figure(figsize=(6, 6))
for line in lines:
    (x_start, y_start), (x_end, y_end) = line
    plt.plot([x_start, x_end], [y_start, y_end], marker='o')

plt.xlim(0, canvas_size[0])
plt.ylim(0, canvas_size[1])
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Generated Lines with Cross-Hatching Pattern')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.show()

# Function to convert generated lines into G-Code
def convert_to_gcode(lines):
    gcode_commands = []
    gcode_commands.append("G21 ; Set units to millimeters")
    gcode_commands.append("G90 ; Use absolute positioning")
    gcode_commands.append("G1 F1500 ; Set feedrate")

    for line in lines:
        (x_start, y_start), (x_end, y_end) = line
        gcode_commands.append(f"G0 X{x_start:.2f} Y{y_start:.2f}")  # Move to start point
        gcode_commands.append(f"G1 X{x_end:.2f} Y{y_end:.2f}")      # Draw line to end point

    return gcode_commands

# Generate G-Code from the lines
gcode_lines = convert_to_gcode(lines)

# Save G-Code to a file
output_file_path = "/Users/lizbethjurado/Keck/Code/generated_lines.gcode"
with open(output_file_path, "w") as gcode_file:
    for command in gcode_lines:
        gcode_file.write(command + "\n")

print(f"G-Code saved to {output_file_path}")
