import numpy as np
import cv2
import argparse

class YOLOPersonDetector:
    def __init__(self, video_path, cfg_path, weights_path):
        self.video_path = '/Users/devarampranith/Desktop/Detection_project/yolo_test.mp4'
        self.cfg_path = '/Users/devarampranith/Desktop/Detection_project/yolov3.cfg'
        self.weights_path = '/Users/devarampranith/Desktop/Detection_project/yolov3.weights'
        self.threshold = 0.7

    def best_box(self, output):
        cordinates = []
        labels = []
        class_prob = []
        for i in output: # each box
            for j in i:  # each box values
                prob_values = j[5:] # 80 classes
                a = np.argmax(prob_values) # pick highest probablity index
                confidence = prob_values[a] # match it with 80 classes
                class_label = 'person'  # Class label for person detection
                if confidence > self.threshold and class_label == 'person':

                    w, h = int(j[2] * 320), int(j[3] * 320)
                    x, y = int(j[0] * 320 - w / 2), int(j[1] * 320 - h / 2)
                    cordinates.append([x, y, w, h])
                    labels.append(a)
                    class_prob.append(confidence)

        final_box = cv2.dnn.NMSBoxes(cordinates, class_prob, self.threshold, 0.6)
        return final_box, cordinates, class_prob, labels

    def prediction(self, final_box, cordinates, probability, names, ow, oh, image):
        for k in final_box.flatten():
            x, y, w, h = cordinates[k]
            x = int(x * ow)
            y = int(y * oh)
            w = int(w * ow)
            h = int(h * oh)
            class_label = names[k]
            acc = str(round(probability[k], 2))

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            text = 'person' + ' ' + str(acc)  # Ensure class_label and acc are strings
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(image, text, (x, y - 2), font, 1, (255, 0, 0), 2)
            cv2.imshow('Detection', image)  # Changed 'frame' to 'image'

    def detect_person(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            neural_network = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)

            while True:
                res, frame = cap.read()
                if res == True:
                    original_height, original_width = frame.shape[:2]
                    input_image_model = cv2.dnn.blobFromImage(frame, 1 / 255, (320, 320), True, crop=False)
                    neural_network.setInput(input_image_model)
                    imp_players = neural_network.getUnconnectedOutLayersNames()
                    output = neural_network.forward(imp_players)
                    final_box, cordinates, probability, names = self.best_box(output)
                    self.prediction(final_box, cordinates, probability, names, original_width / 320,
                                    original_height / 320, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print("An error occurred:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect persons in a video using YOLO.')
    parser.add_argument('video_path', type=str, help='Path to input video file.')
    parser.add_argument('cfg_path', type=str, help='Path to YOLO config file.')
    parser.add_argument('weights_path', type=str, help='Path to YOLO weights file.')
    args = parser.parse_args()

    detector = YOLOPersonDetector(args.video_path, args.cfg_path, args.weights_path)
    detector.detect_person()
