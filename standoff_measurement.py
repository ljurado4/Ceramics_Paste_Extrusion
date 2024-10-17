import cv2
import sys
import statistics
import tkinter
from tkinter import filedialog

def image_import():
    """
    Imports image using CLI or via tkinter dialog window.

    Returns:
        :img (_Matlike_): Array of image pixel color values.
    """    
    if len(sys.argv) > 1:
        # Get the image path from the command-line argument
        image_path = sys.argv[1]
    else:
        # Prevents default tkinter window from opening
        tkinter.Tk().withdraw()

    # Opens native dialog window for image selection
    image_path=filedialog.askopenfilename()
    img = cv2.imread(image_path)

    # Prevent image import error
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    return img

def image_processing(img):
    """
    Applies area and ROI based filters to identify nozzle contour, draws bounding box around it, and sets calibration factor based on it's width.

    Parameters:
        img (_Matlike_): Array of image pixel color values.

    Returns:
        :morph (_Matlike_): Processed image using morphological close to separate binary image regions.
        :width (_int_): Width of full image in pixels.
        :height (_int_): Height of full image in pixels.
        :roi (_Matlike_): Lower half of image to focus on region of interest.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output_path = "image_gray.png"
    cv2.imwrite(output_path, gray)

    # Define region of interest (ROI) - Bottom part of the image
    height, width = gray.shape
    roi = gray[int(height * 0.5):, :]

    # Apply adaptive thresholding to highlight the bright area between the nozzle and the bed
    thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    output_path = "image_thresh.png"
    cv2.imwrite(output_path, thresh)

    # Apply morphological operations to clean up small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Displaying morph for analysis
    output_path = "image_morph.png"
    cv2.imwrite(output_path, morph)

    return morph, width, height, roi

def image_contours(morph, height):
    """
    Applies area based filters to reduce 'noise', i.e. the number of identified contours, and focus on relevant sized ROI.

    Parameters:
        morph (_Matlike_): Processed image using morphological close to separate binary image regions.
        height (_int_): Height of image in pixels.

    Returns:
        :contours (tuple Sequence[Matlike]): All detected closed-body contours.
    """
    # Find contours in the ROI
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a img copy for displaying contour step
    img_contours = img.copy()

    # Area limit for filtering out less significant contours
    min_area = 500
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    # Displaying contours for analysis
    cv2.drawContours(img_contours[int(height * 0.5):], filtered_contours, -1, (0, 255, 0), 2)
    output_path = "image_contour.png"
    cv2.imwrite(output_path, img_contours)

    return contours

def nozzle_contour(contours, width, height):
    """
    Applies area and ROI based filters to identify nozzle contour, draws bounding box around it, and sets calibration factor based on it's width.

    Parameters:
        contours (_tuple Sequence[Matlike]_): All detected closed-body contours.
        width (_int_): Width of image in pixels.
        height (_int_): Height of image in pixels.

    Returns:
        :nozzle_bottom (_tuple [int, int]_): XY coords representing the bottom center of the nozzle contour.
        :calibration_factor (_float_): Ratio for image in mm/pixel.
    """
    # # For use only if multiple sizes are expected
    # # Establishing calibration factor for various nozzle sizes
    # nozzle_ID = float(input("Enter nozzle inner diameter: "))
    # if nozzle_ID == 0.8:
    #     nozzle_OD = 1.26
    nozzle_OD = 1.26

    # Find the lowest point of the nozzle and the highest point of the platform (in the original image)
    nozzle_bottom = None
    nozzle_rect = None

    # Locate the nozzle bottom based on contour analysis
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200:
            x, y, w, h = cv2.boundingRect(contour)
            
            # ROI filter since nozzle should be centered and higher than plate
            if y < height * 0.2 and x<0.75*width and x>0.25*width:
                nozzle_bottom = (x + w // 2, int(height * 0.5) + y + h)
                nozzle_rect = (x, int(height * 0.5) + y, w, h)
    
    # Calculating calibration factor 
    calibration_factor = nozzle_OD / nozzle_rect[2]
    print(f"Calibration Factor = {calibration_factor:.3f} mm/pixel")  

    # Displaying nozzle bounding box as cyan rectangle
    if nozzle_rect is not None:
        cv2.rectangle(img, (nozzle_rect[0], nozzle_rect[1]), 
                        (nozzle_rect[0] + nozzle_rect[2], nozzle_rect[1] + nozzle_rect[3]), 
                        (255, 255, 0), 2)
    else:
        print("Error: Unable to detect nozzle bottom for measurement.")

    return nozzle_bottom, calibration_factor

def plate_contour(contours):
    """
    Applies area and ROI based filters to identify plate contour & draws bounding box around it.

    Parameters:
        contours (_tuple Sequence[Matlike]_): All detected closed-body contours.

    Returns:
        :plate_top (_tuple [int, int]_): XY coords representing the top center of the plate contour.
    """
    # Locate the top of the plate based on contour analysis
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)

            # Shifting ROI to top half of plate contour since longest edges are near top or bottom
            y_coords = [point[0][1] for point in contour]
            sorted_y_coords = sorted(y_coords)
            mid_point = len(sorted_y_coords) // 2
            cutoff_value = sorted_y_coords[mid_point]
            filtered_y_coords = sorted([y for y in y_coords if y < cutoff_value])
            
            try:
                # Calculate mode of y-coordinates to find longest consistent edge
                y = int(statistics.mode(filtered_y_coords))
            except statistics.StatisticsError:
                # Handle cases with no unique mode by using the minimum y-value
                y = int(min(filtered_y_coords))
            plate_top = (x + w // 2, int(height * 0.5) + y)
            plate_rect = (x, int(height * 0.5) + y, w, h)

            # Assuming platform contour is the first prominent one found
            break

    # Displaying plate bounding box as magenta rectangle
    if plate_rect is not None:
        cv2.rectangle(img, (plate_rect[0], plate_rect[1]), 
                        (plate_rect[0] + plate_rect[2], plate_rect[1] + plate_rect[3]), 
                        (255, 0, 255), 2)
    else:
        print("Error: Unable to detect plate top for measurement.")

    return plate_top

def measure_standoff(nozzle_bottom, calibration_factor, plate_top):
    """
    Measures distance between the bounding boxes identified for top bed surface and bottom nozzle surface.

    Parameters:
        image_path (_str_): File path to raw image.
    """    

    try:
        # Calculate the vertical distance between the bottom of the nozzle and the top of the platform (white gap)
        gap_pixels = abs(nozzle_bottom[1] - plate_top[1])
        gap_distance_mm = gap_pixels * calibration_factor

        # Display the result in the terminal
        print(f"Standoff Distance: {gap_distance_mm:.3f} mm")

        # Draw contours for visualization
        img_result = img.copy()
        cv2.circle(img_result, nozzle_bottom, 5, (0, 255, 255), -1)  # Yellow dot for needle bottom
        cv2.circle(img_result, plate_top, 5, (255, 255, 0), -1)  # Cyan dot for platform top

        # Draw the vertical line for distance measurement (white gap)
        x_coord = nozzle_bottom[0]  # Use the x-coordinate of the needle's bottom point
        cv2.line(img_result, (x_coord, nozzle_bottom[1]), (x_coord, plate_top[1]), (0, 0, 255), 2)  # Red vertical line

        # Save or display the resulting image
        output_path = "image_measure.png"
        cv2.imwrite(output_path, img_result)

        # Display the image with the measurements
        # cv2.imshow('Measured Image - White Gap', img_result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except Exception as e:
        print(f"Measurement failed: {e}")

if __name__ == "__main__":
    
    # Import image 
    img = image_import()

    # Adaptive thresholding and noise reduction
    morph, width, height, roi = image_processing(img)

    # Filter contours by area
    contours = image_contours(morph, height)

    # ROI filter to identify lowest horizontal border of nozzle and calibration factor
    nozzle_bottom, calibration_factor = nozzle_contour(contours, width, height)

    # ROI filter to identify highest horizontal border of plate
    plate_top = plate_contour(contours)

    # Measure vertical distance between platform_top and nozzle_bottom and visualize
    measure_standoff(nozzle_bottom, calibration_factor, plate_top)
