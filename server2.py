import base64
import os
import colorsys
import datetime
from collections import Counter
from io import BytesIO

import numpy as np
from http import HTTPStatus
from flask import Flask, request, render_template, Blueprint,\
    jsonify
from PIL import Image

# from histogram.histogram_distance import load_histogram_model, get_similar_by_histogram
# from yolo3.detection import load_model, detect_image, visualize


app = Flask(__name__)

api = Blueprint("api", __name__, template_folder="templates")

# data_directory = "models"
# model_filename = os.path.join(data_directory, "model.h5")
# anchors_filename = os.path.join(data_directory, "anchors.txt")
# classes_filename = os.path.join(data_directory, "classes.txt")

# with open(classes_filename) as f:
#     class_names = f.readlines()

# class_names = [c.strip() for c in class_names]

# with open(anchors_filename) as f:
#     anchors = f.readline()

# anchors = [float(x) for x in anchors.split(",")]
# anchors =  np.array(anchors).reshape(-1, 2)
# model = load_model(model_filename, anchors, class_names)

# hsv_tuples = [
#     (
#         x / len(model.class_names), 1., 1.
#     ) for x in range(len(model.class_names))
# ]

# colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
# colors = list(
#     map(
#         lambda x: (
#             int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)
#         ), colors
#     )
# )

# np.random.seed(1004)
# np.random.shuffle(colors)
# np.random.seed(None)

# hist_data = load_histogram_model(
#     "/Users/williamgunawan/private_workspace/joki/project_histogram/data/histogram_data.pkl"
# )


@api.route("/")
def index():
    return render_template("index2.html")


@api.route("/", methods=["POST"])
def get_data():
    data = request.get_json(force=True)
    if "image_filename_1" not in data:
        return "", HTTPStatus.BAD_REQUEST

    if "image_filename_2" not in data:
        return "", HTTPStatus.BAD_REQUEST

    image_filename_1 = data["image_filename_1"]
    image_filename_2 = data["image_filename_2"]

    return jsonify({
        "detected_image_1": "static/images/image_visualization_205936970289.jpg",
        "detected_image_2": "static/images/image_visualization_205937706757.jpg",
        "histogram_label_images_1": [
            "Lactogen",
            "Frisian Flag",
            "Frisian Flag",
            "Vidoran Xmart",
            "Frisian Flag",
            "Frisian Flag",
            "Frisian Flag",
            "Frisian Flag",
            "Frisian Flag",
            "Vidoran Xmart"
        ],
        "histogram_label_images_2": [
            "Frisian Flag",
            "Frisian Flag",
            "Ensure",
            "Frisian Flag",
            "Ensure",
            "Frisian Flag",
            "Ensure",
            "Vidoran Xmart",
            "Vidoran Xmart"
        ],
        "missing_images": [
            {
                "Lactogen": 1
            },
            {
                "Frisian Flag": 3
            }
        ]
    })

    object_detection_results_1, visualize_image_filename_1 = _get_detected_object(
        image_filename=image_filename_1
    )

    object_detection_results_2, visualize_image_filename_2 = _get_detected_object(
        image_filename=image_filename_2
    )

    list_label_name_1 = []
    for i, x in enumerate(object_detection_results_1):
        image_copy = Image.open(image_filename_1)
        box = x["box_position"]
        image_copy = image_copy.crop(box)

        image_hist = np.array(image_copy)[:, :, ::-1].copy()
        label_name = get_similar_by_histogram(image_hist, hist_data)
        list_label_name_1.append(label_name)

    list_label_name_2 = []
    for i, x in enumerate(object_detection_results_2):
        image_copy = Image.open(image_filename_2)
        box = x["box_position"]
        image_copy = image_copy.crop(box)

        image_hist = np.array(image_copy)[:, :, ::-1].copy()
        label_name = get_similar_by_histogram(image_hist, hist_data)
        list_label_name_2.append(label_name)

    ctr = Counter(list_label_name_1)
    ctr2 = Counter(list_label_name_2)
    ctr2 = {k:(v * -1) for k,v in ctr2.items()}
    ctr.update(ctr2)
    missing_images = [{k:v} for k,v in ctr.items() if v > 0]

    results = {
        "histogram_label_images_1": list_label_name_1,
        "detected_image_1": visualize_image_filename_1,
        "histogram_label_images_2": list_label_name_2,
        "detected_image_2": visualize_image_filename_2,
        "missing_images": missing_images
    }
    return jsonify(results)


def _get_detected_object(image_filename):
    image = Image.open(image_filename)
    object_detection_results = detect_image(image, model)

    # for x in object_detection_results:
    #     print("Label: {}".format(x["label"]))
    #     print("Score: {}".format(x["score"]))
    #     print("Box Coordinate (x1, y1, x2, y2): {}\n".format(x["box_position"]))

    visualize_image = visualize(image, object_detection_results, colors)
    image_filename = image_filename.split("/")[-1]
    visualize_image_filename = "static/images/image_visualization_{}.jpg".format(
        datetime.datetime.now().strftime("%H%M%S%f")
    )
    visualize_image.save(visualize_image_filename)
    return object_detection_results, visualize_image_filename


def get_similar_item(image, hist_data):
    label_name = get_similar_by_histogram(image, hist_data)
    return label_name

app.register_blueprint(api)

if __name__ == "__main__":
    app.run(debug=True)
