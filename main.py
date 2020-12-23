from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as CImage
from clarifai.errors import ApiError
from glob import glob
import os
import cv2

ROOT = "./training/"
app = ClarifaiApp(api_key="7692eedc98ca42108d92aa8460ddbfe2")
MODEL_ID = "facemaskdetection"
CONCEPTS_LIST = ["mask", "nomask", "nose"]


def input_image():
    app.inputs.delete_all()

    folders = CONCEPTS_LIST
    for folder in folders:
        not_concepts = folders.copy()
        not_concepts.remove(folder)
        image_set = create_image_set(ROOT + folder + '/', concepts=[folder], not_concepts=not_concepts)
        app.inputs.bulk_create_images(image_set)

        print(app.inputs.check_status())


def create_model():
    model = app.models.create(MODEL_ID, concepts=CONCEPTS_LIST)


def train():
    model = app.models.get(MODEL_ID)
    model.train()


def create_image_set(path, concepts, not_concepts):
    images = []
    for file in glob(os.path.join(path + '*.jpg')):
        i = CImage(filename=file, concepts=concepts, not_concepts=not_concepts)
        images.append(i)
    return images


def predict_url(url):
    model = app.models.get(MODEL_ID)
    result = model.predict_by_url(url)
    print_prediction(result["outputs"][0]["data"]["concepts"])


def predict_file(file):
    model = app.models.get(MODEL_ID)
    result = model.predict_by_filename(file)
    print_prediction(result["outputs"][0]["data"]["concepts"])


def print_prediction(prediction):
    for concept in prediction:
        print('%s: %f' % (concept['name'], concept['value']))
    print()


def predict_video(file):
    try:
        model = app.models.get(MODEL_ID)
        result = model.predict_by_filename(file)
    except ApiError as e:
        print('Error status code: %d' % e.error_code)
        print('Error description: %s' % e.error_desc)
        if e.error_details:
            print('Error details: %s' % e.error_details)
        exit(1)

    frames = result['outputs'][0]['data']['frames']
    for frame in frames:
        print('Concepts in frame at time: %d ms' % frame['frame_info']['time'])
        for concept in frame['data']['concepts']:
            print('%s: %f' % (concept['name'], concept['value']))
        print()


def capture():
    cam = cv2.VideoCapture(0)

    while True:
        ret, image = cam.read()
        cv2.imshow('image', image)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('cap.jpg', image)
            predict_file("cap.jpg")
        if cv2.waitKey(1) & 0xFF == 27:
            os.remove("cap.jpg")
            break

    cam.release()
    cv2.destroyAllWindows()


capture()
