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
    approx = findOneCardPolygon(img_gray)
    if approx is not None:
        roi = adjustCardPerspective(img_gray, approx)
        class_index = classifyCard(roi)
        if class_index < len(card_list):
            result = card_list[(class_index % 78)]
            result["img_url"] = url_for(
                "static", filename=f'card_images/{result["img"]}'
            )
            return card_list[(class_index % 78)]
    return None


def classifyCard(roi):
    roi = cv2.resize(roi, (180, 180))
    roi = roi.astype("float") / 255.0  # Normalize pixel values

    roi = np.reshape(roi, (1, roi.shape[0], roi.shape[1], 1))

    # Load the saved model from the .h5 file
    model = keras.models.load_model("10epochs_conv.h5")

    # Make predictions using the loaded model
    predictions = model.predict(roi)
    # Decode the predictions (assuming your model output is categorical)
    class_index = np.argmax(predictions) % 78

    return class_index


def getCardJSONInfo(index):
    return card_list[(index % 78)]


def findOneCardPolygon(image):
    # apply binary thresholding
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(
        image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )

    for i, h in enumerate(hierarchy[0]):
        if h[3] == -1:  # contour has no parent
            # Use polygon approximation to simplify the contour
            epsilon = 0.02 * cv2.arcLength(contours[i], True)
            approx = cv2.approxPolyDP(contours[i], epsilon, True)
            area = cv2.contourArea(contours[i])
            if len(approx) == 4 and area > 1000:
                rect = cv2.minAreaRect(approx)
                _, (w, h), _ = rect
                aspect_ratio = max(w, h) / min(w, h)

                if 1.2 < aspect_ratio < 1.8:
                    return approx
    return None


def adjustCardPerspective(image, approx, aspect_ratio=1.7):
    polygon_vertices = [point[0] for point in approx]
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

    # Convert the polygon_vertices to NumPy array
    polygon_vertices = np.array(polygon_vertices, dtype=np.float32)

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(polygon_vertices, target_vertices)

    # Apply the perspective transformation
    roi = cv2.warpPerspective(image, matrix, (int(target_width), int(target_height)))
    return roi


# # Read Image
# image = cv2.imread("testImages/pageofcups.jpg")
# cardJson = oneCardSpread(image)
# # Print the predicted class
# print(f'Predicted Class: {cardJson["name"]}\nCard Meaning: {cardJson["keywords"]}')
