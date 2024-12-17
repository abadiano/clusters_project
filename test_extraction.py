import easyocr
import re
import cv2

def resize_image(image_path, max_width=800):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    if width > max_width:
        scaling_factor = max_width / float(width)
        new_dimensions = (int(width * scaling_factor), int(height * scaling_factor))
        resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_path, resized_image)
    return image_path

def extract_address_easyocr(image_path):
    # Resize image for faster processing
    image_path = resize_image(image_path)

    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Perform OCR on the image
    results = reader.readtext(image_path)

    # Print all
    print("Detected Text Blocks:")
    for bbox, text, confidence in results:
        print(f"Text: {text}, Confidence: {confidence}")

    # Combine all detected text
    address_lines = [text for _, text, _ in results if confidence > 0.5]
    address = " ".join(address_lines)
    return address

if __name__ == "__main__":
    image_path = r"C:\Users\abadianov\Desktop\Projects\testv1.png"
    extracted_address = extract_address_easyocr(image_path)
    print(f"Extracted Address: {extracted_address}")
