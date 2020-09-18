import cv2
from Model import load_weights, image_transforms, tensor_transform, LastLayer_Alexnet, CNNClassifier, newest_model
from PIL import Image
import numpy as np
import time
from Model import classes


class running_majority:
    '''
    this class takes predictions and outputs running majority prediction
    '''

    class TopNHeap:
        """
        A heap that keeps the top N elements around
        """

        def __init__(self, N):
            self.elements = []
            self.N = N

        def add(self, e):
            from heapq import heappush, heapreplace
            if len(self.elements) < self.N:
                heappush(self.elements, e)
            elif self.elements[0] < e:
                heapreplace(self.elements, e)

    def __init__(self, frame_window=10):
        self.h = running_majority.TopNHeap(frame_window)
        self.word_counter = {}
        for c in classes:
            self.word_counter[c] = 0

    def add(self, pred_class):
        self.h.add((float(time.time()), pred_class))

    def predict(self):
        words = [h[1] for h in self.h.elements]
        return max(set(words), key=words.count)


def inference(model, image):
    '''image pre_processing'''
    PIL_image = Image.fromarray(image)  ##convert 2 pil to apply test transforms uniformly
    image_t = tensor_transform['valid'](PIL_image)
    # torchvision.utils.save_image(image_t, "test1.jpg") ##weird this messes things up
    p = model.predict(image_t)
    return p


def run_live_inference(model, camera_source):
    cap = cv2.VideoCapture(camera_source)
    majority = running_majority(frame_window=20)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, image = cap.read()
        image = cv2.resize(image, (256, 200))  ##resize for display
        pred = inference(model, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        majority.add(pred)
        pred = majority.predict()
        print("Detected class: ", pred)
        image = cv2.putText(image, pred, (0, np.shape(image)[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.imshow('stream', image)
        k = cv2.waitKey(1)
        if not ret:
            break
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        # time.sleep(1)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    '''
    We default to most recent model created
    '''
    default_model_path = newest_model()

    parser = argparse.ArgumentParser()
    # Put custom arguments here
    parser.add_argument('-s', '--camera_source', type=int, default=0)
    parser.add_argument('-m', '--model_path', type=str, default=default_model_path)
    args = parser.parse_args()

    print('Using model path: {}'.format(args.model_path))
    model = LastLayer_Alexnet()
    model = load_weights(model, args.model_path)

    run_live_inference(model, args.camera_source)
