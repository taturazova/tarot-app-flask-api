import cv2
import numpy as np
import json
from tensorflow import keras
from flask import url_for

# Read card info
with open("card_data.json", "r") as file:
    card_list = json.load(file)


def oneCardSpread(image):
    img_gray = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_GRAYSCALE)
    img_gray = resizeInputImage(img_gray, 600)
    detectedCard = getOneCardFromImage(img_gray)
    if detectedCard is not None:
        return [str(classifyCard(detectedCard[0]))]
    return []


def classifyCard(roi):
    roi = cv2.resize(roi, (180, 180))
    roi = roi.astype("float") / 255.0  # Normalize pixel values

    roi = np.reshape(roi, (1, roi.shape[0], roi.shape[1], 1))

    # Load the saved model from the .h5 file
    model = keras.models.load_model("25epochs_conv.keras")

    # Make predictions using the loaded model
    predictions = model.predict(roi)
    # Decode the predictions (assuming your model output is categorical)
    class_index = np.argmax(predictions) % 78

    return class_index


def getOneCardFromImage(img_gray):
    # Apply Gaussian Blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(img_gray, (7, 7), 0)

    # Define a sharpening kernel
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Apply the sharpening kernel to the grayscale image
    sharpened = cv2.filter2D(blurred, -1, sharpening_kernel)

    # Use Canny edge detection
    edges = cv2.Canny(sharpened, 150, 255)

    # Find contours on the edges detected image using cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    detectedCard = None

    for i, h in enumerate(hierarchy[0]):
        # Use polygon approximation to simplify the contour

        epsilon = 0.02 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)
        area = cv2.contourArea(contours[i])

        # Calculate the length of the contour
        contour_length = cv2.arcLength(contours[i], True)
        # Calculate the convex hull of the contour
        hull_length = cv2.arcLength(cv2.convexHull(contours[i]), True)

        # Calculate the ratio of contour length to convex hull length
        ratio = contour_length / hull_length if hull_length > 0 else contour_length

        if ratio < 2 and len(approx) == 4 and area > 5000:
            rect = cv2.minAreaRect(approx)
            center, (w, h), angle = rect
            aspect_ratio = max(w, h) / min(w, h)

            polygon_vertices = [point[0] for point in approx]
            # Convert the polygon_vertices to NumPy array
            polygon_vertices = np.array(polygon_vertices, dtype=np.float32)

            polygon_area = cv2.contourArea(polygon_vertices)

            if 1.2 < aspect_ratio < 1.8 and (
                (detectedCard == None or polygon_area > detectedCard[1])
            ):
                # Determine the target width based on the maximum width of the base of the polygon
                target_width = max(
                    np.linalg.norm(polygon_vertices[1] - polygon_vertices[2]),
                    np.linalg.norm(polygon_vertices[3] - polygon_vertices[0]),
                )

                # Calculate the target height based on the aspect ratio
                target_height = int(target_width * aspect_ratio)

                # Define the target rectangle coordinates
                target_vertices = np.array(
                    [
                        [0, 0],
                        [0, target_height],
                        [target_width, target_height],
                        [target_width, 0],
                    ],
                    dtype=np.float32,
                )

                # Compute the perspective transformation matrix
                matrix = cv2.getPerspectiveTransform(polygon_vertices, target_vertices)

                # Apply the perspective transformation
                roi = cv2.warpPerspective(
                    img_gray, matrix, (int(target_width), int(target_height))
                )
                detectedCard = (roi, polygon_area)
    return detectedCard


def resizeInputImage(image, new_height):
    # Get the original dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the new width while maintaining the aspect ratio
    aspect_ratio = original_width / original_height
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    return cv2.resize(image, (new_width, new_height))


def getCardJSONInfo(index):
    return card_list[(index % 78)]


# # # Read Image
# image = cv2.imread("testImages/pageofcups.jpg")
# cardJson = oneCardSpread(image)
# # Print the predicted class
# print(f'Predicted Class: {cardJson["name"]}\nCard Meaning: {cardJson["keywords"]}')
