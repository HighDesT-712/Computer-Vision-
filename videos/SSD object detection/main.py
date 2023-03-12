#VGG16
import cv2
import numpy as np
from libreriascreadas import imagemodel as imo1
from libreriascreadas import filesfunctions as ff1
from libreriascreadas import videofunctions as v1

SSD_INPUT_SIZE = 320
THRESHOLD = 0.6 #50 PERCENT PROBABILITY
SUPPRESSION_THRESHOLD = 0.2 # THE LOWER VALUE THE FEWER BOUNDING BOXES WILL REMAIN

class_names = ff1.construct_class_names() #function defined
#print(type(class_names))
#print(class_names)

#capture = cv2.VideoCapture('objects.mp4')
video_captured  = v1.VideoCaptureInternalWebCam() # desde la camara interna de la laptop

neural_network = cv2.dnn_DetectionModel('ssd_weights.pb','ssd_mobilenet_coco_cfg.pbtxt')
neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


neural_network.setInputSize(SSD_INPUT_SIZE, SSD_INPUT_SIZE)
neural_network.setInputScale(1.0/127.5)
neural_network.setInputMean((127.5,127.5,127.5))
neural_network.setInputSwapRB(True)

while True:
    is_grabbed, frame = video_captured.read()
    if not is_grabbed:
        break
    #bbox is bounding boxes
    class_label_ids, confidences, bbox = neural_network.detect(frame)
    #print(class_label_ids)  # [76]
    #print(type(bbox))

    #we mhave to make sure that this 3 variables are LISTS not numpy arrays.
    bbox =  list(bbox)
    # print(bbox)
    #print(type(bbox))
    #print(confidences)
    #confidences is stores in a 1 dimensional array but the teacher has a 2 dimensional array.
    confidences = np.array(confidences).reshape(1,-1).tolist()[0] #for some reason, [0] is the only approach that works for mi code, altohough my confidence variable
    #is not two dimensional type.
    #print(type(confidences))

    box_to_keep = cv2.dnn.NMSBoxes(bbox,confidences,THRESHOLD, SUPPRESSION_THRESHOLD)
    #print(box_to_keep)
    imo1.show_detected_objectsLevel1(frame, box_to_keep, bbox,class_names,class_label_ids)

    cv2.imshow('SSD Algorithm',frame)
    # si se presiona la tecla ESC, se sale de la reproduccion de video
    key = cv2.waitKey(30) & 0x0ff
    if key == 27:
        break
    #cv2.waitKey(1)

video_captured.release()
cv2.destroyAllWindows()