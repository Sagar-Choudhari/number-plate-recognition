import os
import cv2
import re
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pytesseract
import easyocr

# Load trained model
model = YOLO('model.pt')

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # You can add language codes like ['en', 'hi'] for additional languages


# Function to perform OCR using EasyOCR
def extract_text_with_easyocr(image):
    # Perform OCR using EasyOCR
    ocr_result = reader.readtext(image, detail=0)  # `detail=0` returns only the recognized text
    return " ".join(ocr_result)


# Read cropped images and apply EasyOCR
def read_number_plates_with_easyocr(cropped_dir):
    for filename in os.listdir(cropped_dir):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path1 = os.path.join(cropped_dir, filename)
            cropped_image = cv2.imread(image_path1)

            # display(cropped_image, "Processed Number Plate (EasyOCR)")

            # Perform OCR on the cropped number plate
            detected_text = extract_text_with_easyocr(cropped_image)

            text = filter_plate_characters(detected_text)

            print(f'File: {filename}, Detected Text (EasyOCR): {text}')


# Process the video frame by frame
def process_video(video_path, output_dir='output'):
    cap = cv2.VideoCapture(video_path)

    # Create output directories
    marked_images_dir = os.path.join(output_dir, 'marked_frames')
    cropped_parts_dir = os.path.join(output_dir, 'cropped_parts')
    os.makedirs(marked_images_dir, exist_ok=True)
    os.makedirs(cropped_parts_dir, exist_ok=True)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run the model on the current frame
        results = model(frame)

        # Loop through detected objects (number plates)
        for i, result in enumerate(results):
            boxes = result.boxes  # List of detected boxes
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = box.cls[0].item()  # Class index

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Crop the number plate region from the frame
                cropped_image = frame[y1:y2, x1:x2]

                # Save the cropped image
                cropped_file_path = os.path.join(cropped_parts_dir, f'{frame_count}_{i}_{j}_cropped.png')
                cv2.imwrite(cropped_file_path, cropped_image)

                # Perform OCR on the cropped image
                detected_text = extract_text_from_image(cropped_image)
                print(f'Frame {frame_count}, Detected Text: {detected_text}')

        # Save the frame with detected number plates marked
        marked_frame_path = os.path.join(marked_images_dir, f'marked_frame_{frame_count}.jpg')
        cv2.imwrite(marked_frame_path, frame)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


# Process the video frame by frame with live preview
def process_video_live(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run the model on the current frame
        results = model(frame)

        # Loop through detected objects (number plates)
        for i, result in enumerate(results):
            boxes = result.boxes  # List of detected boxes
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = box.cls[0].item()  # Class index

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Crop the number plate region from the frame
                cropped_image = frame[y1:y2, x1:x2]

                # Perform OCR on the cropped image
                detected_text = extract_text_from_image(cropped_image)

                # Display the detected number plate text on the frame
                cv2.putText(frame, detected_text, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame in a live window
        cv2.imshow('Live Number Plate Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

        print("Frame Count: ", frame_count)

    cap.release()
    cv2.destroyAllWindows()


# Define a function to detect number plates, mark them, and crop them
def process_image(image_path2, save_dir='output'):
    # Load the image
    image = cv2.imread(image_path2)

    # Run the model on the image
    results = model(image)

    # Create folders to save the results if they don't exist
    marked_images_dir = os.path.join(save_dir, 'marked_images')
    cropped_parts_dir = os.path.join(save_dir, 'cropped_parts')
    os.makedirs(marked_images_dir, exist_ok=True)
    os.makedirs(cropped_parts_dir, exist_ok=True)

    # Accessing detected results
    for i, result in enumerate(results):
        boxes = result.boxes  # List of detected boxes

        # Loop through each box
        for j, box in enumerate(boxes):
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box in xyxy format
            conf = box.conf[0].item()  # Confidence score
            cls = box.cls[0].item()  # Class index

            # Draw a rectangle around the detected number plate
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{model.names[int(cls)]} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Crop the number plate part from the image
            cropped_image = image[y1:y2, x1:x2]

            # Save the cropped part
            cropped_file_path = os.path.join(cropped_parts_dir, f'{i}_{j}_cropped.png')
            cv2.imwrite(cropped_file_path, cropped_image)

            # read cropped image using ocr pytesseract
            read_number_plates_from_folder(cropped_parts_dir)

            # read cropped image using easyocr
            read_number_plates_with_easyocr(cropped_parts_dir)

            # read cropped image using google vision

    # Save the image with marked detected objects
    marked_image_path = os.path.join(marked_images_dir, f'marked_{os.path.basename(image_path2)}')
    cv2.imwrite(marked_image_path, image)

    # display(marked_image_path, "Detected number plate")

    print(f'Marked image saved at: {marked_image_path}')
    print(f'Cropped parts saved in: {cropped_parts_dir}')


def display(img_, title=''):
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    ax.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


# Optional: if tesseract is not in PATH, specify the executable path
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

# Preprocess the image for better OCR accuracy
def preprocess_image_for_ocr(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised1 = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    # Apply sharpening filter with the new kernel
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.GaussianBlur(denoised1, (5, 5), 0)

    sharp = cv2.filter2D(filtered, -1, sharpen_kernel)

    # Optionally: Erode and Dilate to remove noise and make text more distinct
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # processed_image = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # Slightly smaller kernel
    # opened = cv2.morphologyEx(denoised2, cv2.MORPH_OPEN, kernel)
    # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    #
    # sharp = cv2.filter2D(closed, -1, sharpen_kernel)

    # Resize the image to make the text more readable for OCR
    height, width = sharp.shape
    # resized_image = cv2.resize(sharp, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    resized_image = cv2.resize(sharp, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)

    # magic_color = apply_brightness_contrast(resized_image, brightness=30, contrast=60)

    # Apply adaptive thresholding to increase contrast
    # thresh = cv2.adaptiveThreshold(resized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY, 11, 2)

    # Apply adaptive thresholding to increase contrast
    # Adjusting blockSize and C constant for finer control
    # thresh = cv2.adaptiveThreshold(resized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY, blockSize=15, C=10)

    # Optionally, you can also experiment with a simpler thresholding method like Otsu's thresholding
    _, otsu_thresh = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # magic_color = apply_brightness_contrast(thresh, brightness=30, contrast=60)

    denoised = cv2.fastNlMeansDenoising(otsu_thresh, None, 30, 7, 21)

    # Example images for illustration (replace with your actual images)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(denoised1, cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    img4 = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
    img6 = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    img7 = cv2.cvtColor(otsu_thresh, cv2.COLOR_BGR2RGB)
    img8 = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

    # List of images and titles
    images = [img, img1, img2, img3, img4, img6, img7, img8]
    titles = ["Original", "Gray", "Denoised1", "Filtered", "Sharp",  "Resized", "Thresh", "Denoised", ]
    rows = 2
    cols = 4
    # Set up the figure size and layout (larger figure size for better display)
    plt.figure(figsize=(15, 8))
    for i in range(len(images)):
        # Add a subplot (rows, cols, index)
        ax = plt.subplot(rows, cols, i + 1)

        # Show image
        ax.imshow(images[i])
        plt.axis('off')  # Turn off the axis
        plt.title(titles[i])
    plt.tight_layout()
    plt.show()

    return denoised


# Perform OCR on the preprocessed image
def extract_text_from_image(image):
    # Preprocess the image for better OCR results
    preprocessed_image = preprocess_image_for_ocr(image)

    # Use pytesseract to do OCR on the preprocessed image
    ocr_result = pytesseract.image_to_string(preprocessed_image, config='--psm 8')

    return ocr_result.strip()


# Function to read cropped images from folder and perform OCR
def read_number_plates_from_folder(cropped_dir):
    for filename in os.listdir(cropped_dir):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path3 = os.path.join(cropped_dir, filename)
            cropped_image = cv2.imread(image_path3)

            # Perform OCR on the cropped number plate
            detected_text = extract_text_from_image(cropped_image)

            text = filter_plate_characters(detected_text)

            print(f'File: {filename}, Detected Text (Tesseract): {text}')


def filter_plate_characters(detected_text):
    """
    Filters the detected text to return only valid vehicle number plate characters.

    Parameters:
        detected_text (str): The detected string from OCR that may contain unwanted characters.

    Returns:
        str: A filtered string containing only uppercase letters and digits.
    """
    # Define the pattern: allow only uppercase letters (A-Z) and digits (0-9)
    pattern = r'[A-Z0-9]+'

    # Convert the detected text to uppercase to ensure uniformity
    detected_text = detected_text.upper()

    # Find all valid sequences of letters and numbers using the pattern
    valid_plate = re.findall(pattern, detected_text)

    # Join the list of valid strings into a single string and return
    return ''.join(valid_plate)


# Example usage
image_path = '/Users/vasundhara-mac/Downloads/numberPlate/car5.jpg'
process_image(image_path)

# video_path = '/Users/vasundhara-mac/Downloads/numberPlate/cars.mp4'
# process_video_live(video_path)
