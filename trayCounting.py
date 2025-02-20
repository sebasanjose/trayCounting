import cv2
import numpy as np

# Define Regions of Interest (ROIs) for the three compartments.
# These coordinates (x, y, width, height) should be calibrated to your tray.
compartment_rois = {
    "compartment1": (90, 50, 350, 550),   # Adjust these values as needed
    "compartment2": (500, 50, 200, 550),
    "compartment3": (720, 50, 320, 550)
}

# Set the low threshold for triggering an alert.
LOW_THRESHOLD = 1

def process_roi(roi):
    """
    Process a Region of Interest to count empanadas using contour detection.
    """
    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Threshold the image; adjust the threshold value based on lighting and empanada color.
    ret, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    count = 0
    # Filter contours based on area to avoid counting noise
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:  # This area threshold may need tuning
            valid_contours.append(cnt)
            count += 1
    return count, valid_contours

def main():
    # Open the webcam (0 is typically the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        output_frame = frame.copy()
        alerts = []
        
        # Process each compartment
        for name, (x, y, w, h) in compartment_rois.items():
            roi = frame[y:y+h, x:x+w]
            count, contours = process_roi(roi)
            
            # Draw the compartment ROI rectangle on the output frame
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Display the count on the frame
            cv2.putText(output_frame, f"{name}: {count}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw bounding boxes for each detected empanada within the ROI
            for cnt in contours:
                # Get bounding rectangle for the contour
                (cx, cy, cw, ch) = cv2.boundingRect(cnt)
                # Draw the bounding box, adjusted for the ROI offset
                cv2.rectangle(output_frame, (x + cx, y + cy), (x + cx + cw, y + cy + ch), (0, 0, 255), 2)
            
            # Check if count is below threshold and trigger an alert
            if count < LOW_THRESHOLD:
                alerts.append(f"{name} is running low!")
                cv2.putText(output_frame, "LOW!", (x, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Print alerts to the console
        for alert in alerts:
            print(alert)
        
        # Show the video feed with annotations
        cv2.imshow("Empanada Tray Monitor", output_frame)
        
        # Exit when the ESC key is pressed
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()