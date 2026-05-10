from flask import Flask, jsonify, request
import requests
import numpy as np
import cv2
import mediapipe as mp
import base64
import os

app = Flask(__name__)

SEGMENT_SECRET = os.getenv("SEGMENT_SECRET")

mp_selfie_segmentation = mp.solutions.selfie_segmentation


@app.route("/health")
def health():
    return jsonify({
        "ok": True
    })


def check_auth():
    if not SEGMENT_SECRET:
        return False, "SEGMENT_SECRET is not configured"

    auth_header = request.headers.get("Authorization")

    if not auth_header:
        return False, "Missing authorization header"

    expected = f"Bearer {SEGMENT_SECRET}"

    if auth_header != expected:
        return False, "Unauthorized"

    return True, None


@app.route("/segment", methods=["POST"])
def segment():
    is_authorized, auth_error = check_auth()

    if not is_authorized:
        return jsonify({
            "error": auth_error
        }), 401

    body = request.get_json(silent=True)

    if not body:
        return jsonify({
            "error": "Invalid JSON body"
        }), 400

    image_url = body.get("image_url")

    if not image_url:
        return jsonify({
            "error": "image_url required"
        }), 422

    try:
        response = requests.get(image_url, timeout=15)

        if response.status_code != 200:
            return jsonify({
                "error": "Failed to fetch image",
                "status_code": response.status_code
            }), 400

        image_bytes = response.content

        image_array = np.frombuffer(image_bytes, np.uint8)

        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                "error": "Failed to decode image"
            }), 400

        height, width = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        ) as segmenter:
            results = segmenter.process(rgb)

            mask = results.segmentation_mask

            if mask is None:
                return jsonify({
                    "error": "No segmentation mask returned"
                }), 422

            binary_mask = (mask > 0.6).astype(np.uint8) * 255

        eyebrow_y = body.get("eyebrow_y")

        eyebrow_y_used = None

        if eyebrow_y is not None:
            try:
                eyebrow_y_used = int(eyebrow_y)
                eyebrow_y_used = max(0, min(eyebrow_y_used, height))

                binary_mask[eyebrow_y_used:, :] = 0
            except ValueError:
                return jsonify({
                    "error": "eyebrow_y must be a number"
                }), 422

        close_kernel = np.ones((15, 15), np.uint8)
        open_kernel = np.ones((5, 5), np.uint8)

        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_CLOSE,
            close_kernel
        )

        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_OPEN,
            open_kernel
        )

        success, png_buffer = cv2.imencode(".png", binary_mask)

        if not success:
            return jsonify({
                "error": "Failed to encode mask"
            }), 500

        mask_base64 = base64.b64encode(png_buffer).decode("utf-8")

        return jsonify({
            "success": True,
            "width": width,
            "height": height,
            "eyebrow_y_used": eyebrow_y_used,
            "mask_base64": mask_base64
        })

    except requests.exceptions.Timeout:
        return jsonify({
            "error": "Image fetch timed out"
        }), 408

    except requests.exceptions.RequestException as error:
        return jsonify({
            "error": f"Image fetch failed: {str(error)}"
        }), 400

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)