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

df = pd.DataFrame(detection, columns =['name', 'percentage_probability', 'box_points']) 
df = df.drop(['box_points'], axis = 1)
df = df[df.name == 'traffic light']
num_trafficLight = len(df)
max_df = df[df.percentage_probability == df.percentage_probability.max()]
max_score = max_df.iat[0,1]
print(type(max_score))
print(num_trafficLight)

def isTrafficLight(num_trafficLight, max_score):
	if num_trafficLight > 0 and max_score >= 60.00:
		print('There is a traffic light')
		return True
	else: 
		print("No traffic light")
		return False



isTrafficLight(num_trafficLight, max_score)

