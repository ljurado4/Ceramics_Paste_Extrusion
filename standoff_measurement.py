import cv2
import sys
import statistics
import tkinter
from tkinter import filedialog
from google.cloud import vision
import io

# Import image using CLI or file dialog
def image_import():
    if len(sys.argv) > 1:
        # Get image from command-line
        image_path = sys.argv[1]
    else:
        tkinter.Tk().withdraw()  # Stop tkinter from opening a window
        image_path = filedialog.askopenfilename()  # Open file dialog
    img = cv2.imread(image_path)  # Read the image

    if img is None:
        print(f"Error: Couldn't load image at {image_path}")
        return
    return img, image_path

# Use Google Vision API to find objects in the image
def google_vision_boundary_analysis(image_path):
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.object_localization(image=image)

    contours = []
    for obj in response.localized_object_annotations:
        print(f"Object: {obj.name}, Confidence: {obj.score}")
        box = [(int(vertex.x * 1000), int(vertex.y * 1000)) for vertex in obj.bounding_poly.normalized_vertices]
        contours.append(box)
    return contours

# Preprocess the image for better analysis
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    height, width = gray.shape
    roi = gray[int(height * 0.5):, :]  # Focus on the bottom part
    thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)  # Threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Clean up noise
    return morph, width, height, roi

# Find nozzle using Google Vision data
def find_nozzle(contours, width, height):
    nozzle_bottom = None
    nozzle_rect = None
    nozzle_OD = 1.26  # Nozzle outer diameter in mm

    for contour in contours:
        x = min([point[0] for point in contour])
        y = min([point[1] for point in contour])
        w = max([point[0] for point in contour]) - x
        h = max([point[1] for point in contour]) - y

        if y < height * 0.2 and x < 0.75 * width and x > 0.25 * width:
            nozzle_bottom = (x + w // 2, int(height * 0.5) + y + h)
            nozzle_rect = (x, int(height * 0.5) + y, w, h)

    if nozzle_rect:
        calibration_factor = nozzle_OD / nozzle_rect[2]  # Calculate mm/pixel
        print(f"Calibration Factor: {calibration_factor:.3f} mm/pixel")
        return nozzle_bottom, calibration_factor
    else:
        print("No nozzle detected!")
        return None, None

# Find the top of the plate
def find_plate(contours):
    for contour in contours:
        x = min([point[0] for point in contour])
        y = min([point[1] for point in contour])
        w = max([point[0] for point in contour]) - x
        h = max([point[1] for point in contour]) - y
        plate_top = (x + w // 2, int(y))
        return plate_top

# Measure distance between nozzle and plate
def measure_standoff(nozzle_bottom, calibration_factor, plate_top):
    if nozzle_bottom and plate_top:
        gap_pixels = abs(nozzle_bottom[1] - plate_top[1])  # Get pixel gap
        gap_distance_mm = gap_pixels * calibration_factor  # Convert to mm
        print(f"Standoff Distance: {gap_distance_mm:.3f} mm")
    else:
        print("Couldn't measure standoff distance!")

if __name__ == "__main__":
    # Import image
    img, image_path = image_import()

    # Preprocess the image
    morph, width, height, roi = preprocess_image(img)

    # Use Google Vision API to detect objects
    contours = google_vision_boundary_analysis(image_path)

    # Find the nozzle position
    nozzle_bottom, calibration_factor = find_nozzle(contours, width, height)

    # Find the plate position
    plate_top = find_plate(contours)

    # Measure and display the standoff distance
    measure_standoff(nozzle_bottom, calibration_factor, plate_top)
