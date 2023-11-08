
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def error_level_analysis(image_bytes, quality=90, contour_value=150):
    nparr = np.fromstring(image_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    _, encoded_image = cv2.imencode('.jpg', original_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    decoded_image = cv2.imdecode(encoded_image, 1)

    ela_image = cv2.absdiff(original_image, decoded_image)
    scale = 15  # The scale factor to enhance differences
    ela_image = cv2.cvtColor(ela_image, cv2.COLOR_BGR2GRAY)
    ela_image = np.array(ela_image * scale, dtype=np.uint8)

    threshold_value = np.max(ela_image) * 0.2  # Thresholding value to identify significant differences
    _, ela_mask = cv2.threshold(ela_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours and draw bounding box
    contours, _ = cv2.findContours(ela_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > contour_value:  # Filter out small areas
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return original_image

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Perform ELA on an image to detect forgeries.')
#     parser.add_argument('image_path', type=str, help='Path to the image file')
#     parser.add_argument('--quality', type=int, default=90, help='JPEG encoding quality for ELA')
#     parser.add_argument('--save', type=str, help='Path to save the output image, if any')
#     parser.add_argument('--show', action='store_true', help='Show the result image using matplotlib')

#     args = parser.parse_args()

#     result_image = error_level_analysis(args.image_path, args.quality)

#     if args.save:
#         cv2.imwrite(args.save, result_image)

#     if args.show:
#         plt.figure(figsize=(10, 10))
#         plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
#         plt.title('Detected Forgeries')
#         plt.axis('off')
#         plt.show()