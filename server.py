from flask import Flask
import yolov5
app = Flask(__name__)

# load model
model = yolov5.load('keremberke/yolov5m-garbage')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# set image
img = 'http://192.168.173.239/capture'

# perform inference
# results = model(img, size=640)
#
# # inference with test time augmentation
# results = model(img, augment=True)
#
# # parse results
# predictions = results.pred[0]
# boxes = predictions[:, :4]  # x1, y1, x2, y2
# scores = predictions[:, 4]
# categories = predictions[:, 5]
#
# what = results.names[int(categories[0])]
# print(what)
# # show detection bounding boxes on image
# results.show()
#
# # save results into "results/" folder
# results.save(save_dir='results/')

@app.route("/process")
def process():
    # perform inference
    # results = model(img, size=640)

    # inference with test time augmentation
    results = model(img, augment=True)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    res_out = results.names[int(categories[0])] if len(categories) > 0 else "error"
    if res_out == "biodegradable":
        return "1"
    elif res_out == "paper":
        return "1"
    elif res_out == "cardboard":
        return "1"
    elif res_out == "glass":
        return "2"
    elif res_out == "metal":
        return "2"
    elif res_out == "plastic":
        return "3"
    else:
        return "0"

if __name__ == "__main__":
    app.run("0.0.0.0")