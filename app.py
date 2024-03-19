from flask import Flask, request, jsonify
from tarot_cards_detect import oneCardSpread

app = Flask(__name__)


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

        # Call the tarot card detection method
        result = oneCardSpread(image)

        if result is not None:
            # Tarot card found
            return jsonify(result), 200
        else:
            # Tarot card not found
            return jsonify(None), 200

    except Exception as e:
        # Handle errors
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
