import io
from flask import Flask, request, jsonify
from PIL import Image
from pillow_heif import register_heif_opener
from tarot_cards_detect import oneCardSpread, threeCardSpread
from request_methods import process_card_request
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)


@app.route("/one_card_spread", methods=["POST"])
def one_card_spread():
    return process_card_request(oneCardSpread)


@app.route("/three_card_spread", methods=["POST"])
def three_cards_spread():
    return process_card_request(threeCardSpread)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
