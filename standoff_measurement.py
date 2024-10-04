import cv2  # Ensure OpenCV is installed
import numpy as np
import sys

def measure_white_gap(image_path):
    """
    Calculate and display the vertical distance of the white gap between the bottom of the nozzle and the top of the platform.
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load image at {image_path}")
            return

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Define region of interest (ROI) - Bottom part of the image
        height, width = gray.shape
        roi = gray[int(height * 0.5):, :]  # Bottom half of the image

        # Apply adaptive thresholding to highlight the bright areas (white gap)
        thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Apply morphological operations to clean up small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours in the ROI
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the lowest point of the nozzle and the highest point of the platform (in the original image)
        needle_bottom = None
        platform_top = None

        # Locate the needle bottom based on contour analysis
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # Adjust threshold to focus on significant contours only
                x, y, w, h = cv2.boundingRect(contour)
                if y < height * 0.2:  # Should be closer to the top of ROI
                    needle_bottom = (x + w // 2, int(height * 0.5) + y + h)

        # Locate the top of the platform (white gap)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Adjust threshold for a prominent platform
                x, y, w, h = cv2.boundingRect(contour)
                platform_top = (x + w // 2, int(height * 0.5) + y)

                # Assuming platform contour is the first prominent one found
                break

        # Check if both points were detected properly
        if needle_bottom is None or platform_top is None:
            print("Error: Unable to detect both the nozzle bottom and platform top for measurement.")
            return

        # Calculate the vertical distance between the bottom of the nozzle and the top of the platform (white gap)
        gap_pixels = abs(needle_bottom[1] - platform_top[1])

        # Assuming a known calibration factor: 1 pixel = 0.1 mm (this should be calibrated)
        calibration_factor = 0.1  # mm per pixel
        gap_distance_mm = gap_pixels * calibration_factor

        # Display the result in the terminal
        print(f"Vertical White Gap Distance: {gap_distance_mm:.2f} mm")

        # Draw contours for visualization
        img_result = img.copy()
        cv2.circle(img_result, needle_bottom, 5, (0, 255, 255), -1)  # Yellow dot for needle bottom
        cv2.circle(img_result, platform_top, 5, (255, 255, 0), -1)  # Cyan dot for platform top

        # Draw the vertical line for distance measurement (white gap)
        x_coord = needle_bottom[0]  # Use the x-coordinate of the needle's bottom point
        cv2.line(img_result, (x_coord, needle_bottom[1]), (x_coord, platform_top[1]), (0, 0, 255), 2)  # Red vertical line

        # Save or display the resulting image
        output_path = "measured_image_white_gap_v2.png"
        cv2.imwrite(output_path, img_result)
        print(f"Result image saved as {output_path}")

        # Display the image with the measurements
        cv2.imshow('Measured Image - White Gap', img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Measurement failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Get the image path from the command-line argument
        image_path = sys.argv[1]
    else:
        # Ask the user to input the image path if not provided
        image_path = input("Enter the path to the image: ")

    # Call the measurement function with the provided image path
    measure_white_gap(image_path)
