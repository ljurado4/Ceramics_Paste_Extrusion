# Ceramics_Paste_Extrusion

## Current Standoff Distance Measurement for Precise 3D Printing ​ 
Using image processing techniques, including OpenCV, to detect the nozzle's contours and the plate used in 3D printing. The primary objective is to measure the vertical standoff distance between the nozzle and the plate. Current functionality detects discrepancies between manual and automated measurements, and future improvements focus on adding a fixed reference point for recalibration to enhance accuracy.

### Features
### `standoff_distance.py`
- **OpenCV Implementation**: Detects contours of the nozzle and the plate from an image.
- **Measurement Discrepancies**: Compares automatic measurements to manual ones, identifying discrepancies.
- **Future Enhancements**:
  - Introduce a fixed reference point using known dimensions of the nozzle.
  - Recalibrate measurements based on this fixed reference to improve accuracy.


## Slicing Development 
Implement a slicing functionality for 3D models, particularly STL files, by generating perimeter points for each layer. The script uses various techniques to read STL files, produce a viewing window of the 3D figure, and slice the object into layers with points generated in a specific order. Future development goals include handling more complex geometries and improving the slicing logic.


### Features
### 'slicing_infill.py'
- **File Reading**: Processes STL files to extract 3D geometry.
- **View Generation**: Renders a viewing window to display the 3D figure.
- **Object Slicing**:
   - Identifies perimeter vertices of the object in a clockwise direction.
   - Generates points between these vertices based on an offset.
   - Determines starting points per layer and records points in tuple form (x, y) in the generation order.

- **Infill Patterns**: Added support for multiple infill patterns with selection via character or number input:
   - 1 or A - Linear Infill: Creates a continuous, snake-like zigzag pattern that moves back and forth across the layer. Ideal for simple shapes and quick printing.
   - 2 or B - Cross Hatching Infill: Generates a cross-hatching effect with alternating lines for improved strength and rigidity.
   - 3 or C - Radial Infill: Fills the layer from the outer perimeter inward, creating a concentric pattern that distributes stress evenly.
   - 4 or D - Adaptive Spiral Infill: Draws a smooth, continuous inward spiral, adjusting to the shape’s boundary and providing consistent material distribution for complex shapes.

- **Future Enhancements**:
Improve starting point logic for complex geometries.
Enhance file input handling and develop a GUI for easier selection of infill patterns and slicing parameters


## Branching Guidelines

When contributing new features or fixing bugs in this repository, follow these branch naming conventions to maintain clarity and organization:

**For Bug Fixes**:  
   Use the prefix `fix/` followed by a short description of the fix. Example:
   ```bash
   git checkout -b fix/fix-out-of-range-error
   ```
**For New Features**:  
   Use the prefix `feature/` followed by a short description of the feature being implemented. Example:
   ```bash
   git checkout -b feature/implement-infill-patterns
   ```
**For Enhancements or Refactors**:  
   Use the prefix `enhancement/` followed by a description. Example:
   ```bash
   git checkout -b enhancement/improve-gui-functionality
   ```
**For Hotfixes** (Urgent production fixes):
   Use the prefix `hotfix/` followed by a description. Example:
   ```bash
   git checkout -b hotfix/fix-crash-on-startup
   ```

## Getting Started

### Clone the Repository

To clone the repository onto your local machine, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a New Branch**:
   Once inside the project directory, create a new branch for your work:
   ```bash
   git checkout -b <branch-name>
   ```

   For example, to create a branch for a new feature:
   ```bash
   git checkout -b feature/add-slicing-visualization
   ```

### Install Required Dependencies

Ensure you have the required libraries installed, such as OpenCV and matplotlib. Install them using:
```bash
pip install -r requirements.txt
```

### Running the Scripts

- **To run the slicing script**:
  ```bash
  python slicing.py
  ```

- **To run the standoff distance measurement script**:
  ```bash
  python standoff_distance.py
  ```

## Contributing

We welcome contributions! Please ensure that you:

1. Follow the branch naming conventions.
2. Write clear and descriptive commit messages.
3. Test your code thoroughly before submitting a pull request.
