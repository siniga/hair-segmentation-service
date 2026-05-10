from flask import Flask, jsonify, request
import requests
import numpy as np
import cv2
import mediapipe as mp
import base64

app = Flask(__name__)

mp_selfie_segmentation = mp.solutions.selfie_segmentation


@app.route("/health")
def health():
    return jsonify({
        "ok": True
    })


@app.route("/segment", methods=["POST"])
def segment():

    body = request.json

    image_url = body.get("image_url")

    if not image_url:
        return jsonify({
            "error": "image_url required"
        }), 422

    try:

        # DOWNLOAD IMAGE
        response = requests.get(image_url)

        image_bytes = response.content

        image_array = np.frombuffer(image_bytes, np.uint8)

        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                "error": "Failed to decode image"
            }), 400

        height, width = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MEDIAPIPE SEGMENTATION
        with mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        ) as segmenter:

            results = segmenter.process(rgb)

            mask = results.segmentation_mask

            binary_mask = (mask > 0.5).astype(np.uint8) * 255

        # CONVERT MASK TO PNG
        _, png_buffer = cv2.imencode(".png", binary_mask)

        mask_base64 = base64.b64encode(png_buffer).decode("utf-8")

        return jsonify({
            "success": True,
            "width": width,
            "height": height,
            "mask_base64": mask_base64
        })

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)