from PIL import Image  # Python Imaging Library for handling images
import pytesseract  # Tesseract OCR library for extracting text from images

def extract_text_from_image(image_path: str) -> str:
    """
    Extracts text from an image file using Tesseract OCR.
    
    This function opens the image from the specified path and uses Tesseract 
    OCR with a specific configuration to extract the text contained within the image.

    Parameters:
    image_path (str): The file path to the image from which text will be extracted.

    Returns:
    str: The extracted text from the image.
    """
    # Open the image file specified by the image_path
    image = Image.open(image_path)

    # Define custom Tesseract OCR configuration
    # --oem 3: Use the LSTM-based OCR engine
    # --psm 1: Automatic page segmentation with orientation and script detection (OSD)
    custom_config = r'--oem 3 --psm 1'

    # Extract text from the image using Tesseract OCR with the specified configuration
    text = pytesseract.image_to_string(image, config=custom_config)
    
    return text


def main() -> None:
    """
    Placeholder for the main logic of the script.
    
    :raises NotImplementedError: This function is not implemented yet.
    """
    raise NotImplementedError("Main testing not implemented")


if __name__ == "__main__":
    main()
