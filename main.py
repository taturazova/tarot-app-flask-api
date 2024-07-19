import io
from flask import Flask, request, jsonify
from PIL import Image
from pillow_heif import register_heif_opener
from tarot_cards_detect import oneCardSpread
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"heic", "jpg", "jpeg", "png"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/one_card_spread", methods=["POST"])
def one_card_spread():
    try:
        # Check if the request contains the 'image' file
        if "image" not in request.files:
            return jsonify({"error": "No 'image' file in the request"}), 400

        # Get the grayscale image from the request
        image = request.files["image"]

        # Check if the file is empty
        if image.filename == "":
            return jsonify({"error": "'image' file name is empty"}), 400

        # Check if the file format is allowed
        if not allowed_file(image.filename):
            return (
                jsonify(
                    {
                        "error": "Unsupported file format. Supported formats are: .heic, .jpg, .jpeg, .png"
                    }
                ),
                400,
            )

        # Read the image data
        image_data = image.read()

        # Convert HEIC image to JPEG format if necessary
        if image.filename.lower().endswith(".heic"):
            register_heif_opener()
            img = Image.open(io.BytesIO(image_data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            image_data = img_bytes.read()

        # Call the tarot card detection method
        detected_cards = oneCardSpread(image_data)
        result = {"detected_card_ids": detected_cards}  # Create the response dictionary
        # Log request result
        logging.info(f"Request result: {result}")

        if result is not None:
            # Tarot card found
            return jsonify(result), 200
        else:
            # Tarot card not found
            return jsonify(None), 200

    except Exception as e:
        # Handle errors
        # Log request result
        logging.info(f"Request ERROR: {e}")

        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
