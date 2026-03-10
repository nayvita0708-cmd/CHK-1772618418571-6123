import cv2
import numpy as np
from PIL import Image, ImageChops
import os

def perform_ela(image_path, quality=90, scale_multiplier=15):
    """
    Performs Error Level Analysis (ELA) on an image.
    """
    try:
        # 1. Open the original image using PIL
        original_img = Image.open(image_path).convert('RGB')
        
        # 2. Save a temporary compressed version
        temp_filename = 'temp_compressed.jpg'
        original_img.save(temp_filename, 'JPEG', quality=quality)
        
        # 3. Open the temporary compressed image
        compressed_img = Image.open(temp_filename)
        
        # 4. Calculate the absolute difference between the original and compressed
        # ImageChops.difference subtracts the pixel values. 
        # Areas that compress differently (like AI manipulated faces) will stand out.
        ela_image = ImageChops.difference(original_img, compressed_img)
        
        # 5. Get the maximum pixel value from the difference image to calculate scaling
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        
        # Prevent division by zero if the images are completely identical
        if max_diff == 0:
            max_diff = 1
            
        # 6. Enhance the brightness of the difference image so we can actually see it
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale * scale_multiplier)
        
        # Clean up the temporary file
        os.remove(temp_filename)
        
        return ela_image

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# --- How to test it ---
if __name__ == "__main__":
    from PIL import ImageEnhance # Import here for the enhancement step
    
    # Replace this with the path to an image you want to test
    test_image_path = "test_photo.jpg" 
    
    print(f"Running Error Level Analysis on {test_image_path}...")
    
    if os.path.exists(test_image_path):
        result_img = perform_ela(test_image_path)
        
        if result_img:
            # Convert PIL image back to OpenCV format (numpy array) to display it
            result_cv = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)
            
            # Show the original and the ELA result
            original_cv = cv2.imread(test_image_path)
            cv2.imshow("Original Image", original_cv)
            
            cv2.imshow("ELA Result (Look for bright, unnatural glowing areas)", result_cv)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Please provide a valid image path to test.")