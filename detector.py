from tensorflow import keras
from imageai.Detection import ObjectDetection
import pandas as pd
detector = ObjectDetection()

model_path = "./models/yolo-tiny.h5"
input_path = "./input/housing_1.jpg"
output_path = "./output/newimage2.jpg"

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)
df = pd.DataFrame()
test = []
for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])
    if eachItem["name"] == "traffic light" and eachItem["percentage_probability"] >= 0.50:
    	test.append(eachItem["name"])

print(test)


