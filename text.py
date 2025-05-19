import cv2
import numpy as np

# Load user image
user_img = cv2.imread("girl-front.jpg")
# Load shirt image with transparency (PNG)
shirt_img = cv2.imread("shirt.png", cv2.IMREAD_UNCHANGED)

# Check if images are loaded properly
if user_img is None or shirt_img is None:
    print("Error: Could not load user image or shirt image.")
    exit()

# Resize shirt image proportionally based on user image width
shirt_width = int(user_img.shape[1] * 0.3)  # Resize shirt width to 30% of user image width
shirt_height = int(shirt_width * shirt_img.shape[0] / shirt_img.shape[1])  # Maintain aspect ratio
shirt_resized = cv2.resize(shirt_img, (shirt_width, shirt_height))

# Define region for overlay (you can adjust these values as needed)
x_offset, y_offset = 180, 200

# Check if shirt image goes out of bounds
if (y_offset + shirt_resized.shape[0] > user_img.shape[0]) or \
   (x_offset + shirt_resized.shape[1] > user_img.shape[1]):
    print("Error: Shirt goes out of bounds. Adjust x_offset/y_offset or resize.")
    exit()

# Split the shirt image into RGB and alpha channels
b, g, r, a = cv2.split(shirt_resized)
shirt_rgb = cv2.merge((b, g, r))
mask = cv2.merge((a, a, a)) / 255.0

# Define the region of interest (ROI) in the user image where the shirt will be placed
roi = user_img[y_offset:y_offset+shirt_resized.shape[0], x_offset:x_offset+shirt_resized.shape[1]]

# Blend the shirt onto the user image within the ROI
blended = (roi * (1 - mask) + shirt_rgb * mask).astype(np.uint8)
user_img[y_offset:y_offset+shirt_resized.shape[0], x_offset:x_offset+shirt_resized.shape[1]] = blended

# Show the final result
cv2.imshow("Virtual Try-On", user_img)
cv2.imwrite("output_tryon.jpg", user_img)  # Save the output image
cv2.waitKey(0)
cv2.destroyAllWindows()
