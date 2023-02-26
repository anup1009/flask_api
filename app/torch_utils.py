import PIL
import keras
import numpy as np
from keras.models import model_from_json
import math

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("weights.h5")

def get_prediction(img):
    height = 480 - img.shape[0]
    width = 480 - img.shape[1]

    if(height % 2 == 1 & width % 2 == 1):
        height1,height2 = math.floor(height/2), math.floor(height/2) + 1
        width1,width2 = math.floor(width/2), math.floor(width/2) +1
    elif(width % 2 == 1):
        width1,width2 = math.floor(width/2), math.floor(height/2) + 1
        height1,height2 = int(height/2), int(height/2)
    elif(height % 2 == 1):
        height1,height2 = math.floor(height/2), math.floor(height/2) + 1
        width1,width2 = int(width/2), int(width/2) 
    else:
        height1,height2 = int(height/2), int(height/2)
        width1,width2 = int(width/2), int(width/2)

    if(height == 0):
        img = np.lib.pad(img, ((0,0),(width1, width2),(0,0)), 'edge')
    elif (width == 0):
        img = np.lib.pad(img, ((height1, height2),(0,0),(0,0)), 'edge')
    else:
        img = np.lib.pad(img, ((height1, height2),(width1, width2),(0,0)), 'edge')
    
    x = img
    x = x.astype(np.uint8)
    x = x/ 255.0
    x.shape = (1, ) + x.shape
    p = model.predict(x)[0,0].round(2)
    return str(p)

    
