#import the object detection library from the imageAI dependency.
from imageai.Detection import ObjectDetection
#import the os module since we'll be dealing with paths and directories.
import os

execution_path = os.getcwd()

# set the ObjectDetection to a variable called detector
detector = ObjectDetection()
# we'll be using the RetinaNet model
# call the RetinaNet
detector.setModelTypeAsRetinaNet()
# set the path of execution
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))

# Call the loadmodel function
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))


#loop through the whole image and identify all the objects and their percentage probability.
for eachObject in detections:
    print(eachObject["name"] + " : " + eachObject["percentage_probability"] )