# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
#from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
#from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
#import cv2
import h5py
import os
import json
import datetime
import time
from logger.logger import logger

def load_configs(model_config_path):
    #model_config='conf/conf_xception.json'
    with open(model_config_path) as f:    
        config = json.load(f)
    logger.info("config:%s",config)
    # config variables
    model_name 		= config["model"]
    weights 		= config["weights"]
    include_top 	= config["include_top"]
    test_size 		= config["test_size"]
    #results 		= config["results"]
    train_path 		= config["train_path"]   #training data path
    features_path 	= config["features_path"]
    labels_path 	= config["labels_path"]
    model_path 		= config["model_path"]
    # path to training dataset
    train_labels = os.listdir(train_path)
    
    
    response_json={"model_config_path":model_config_path,
                   "model_name":model_name,"weights":weights,
                  "include_top":include_top,"test_size":test_size,
                  "train_path":train_path,"features_path":features_path,
                  "labels_path":labels_path,"model_path":model_path,
                   "train_labels":train_labels}
    response_json = json.dumps(response_json) #convert in json string obj
    return response_json

def select_model(model_name,weights):
    logger.info(" loading base model and model...")
    # create the pretrained models
    # check for pretrained weight usage or not
    # check for top layers to be included or not
    if model_name == "vgg16":
        base_model = VGG16(weights=weights)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        image_size = (224, 224)
    elif model_name == "vgg19":
        base_model = VGG19(weights=weights)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        image_size = (224, 224)
    elif model_name == "resnet50":
        base_model = ResNet50(weights=weights)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
        image_size = (224, 224)
    elif model_name == "inceptionv3":
        base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('custom').output)
        image_size = (299, 299)

    elif model_name == "inceptionresnetv2":
        base_model = InceptionResNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('custom').output)
        image_size = (299, 299)
    elif model_name == "mobilenet":
        base_model = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('custom').output)
        image_size = (224, 224)

    elif model_name == "xception":
        base_model = Xception(weights=weights)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        image_size = (299, 299)
    else:
        base_model = None

    logger.info(" successfully loaded base model and model...")
    response_json={"base_model":base_model,"model":model,"image_size":image_size}
    #response_json = json.dumps(response_json) #convert in json string obj
    #model_list=[base_model,model,image_size]
    return response_json
def extract_features(model_path,model,train_path,image_size):
    # start time
    print ("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    start = time.time()
    
    train_labels = os.listdir(train_path)
    # encode the labels
    print ("[INFO] encoding labels...")
    le = LabelEncoder()
    le.fit([tl for tl in train_labels])

    # variables to hold features and labels
    features = []
    labels   = []
    start_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M") #datetime.datetime.now()
    logger.debug("[STATUS] start time - %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    # loop over all the labels in the folder
    count = 1
    try:
        for i, label in enumerate(train_labels):
            cur_path = train_path + "/" + label + "/images" #NIH
            #cur_path = train_path + "/" + label   #flower17
            count = 1
            #for image_path in glob.glob(cur_path + "/*.png"): #NIH - png,  #flower16- jpg
            for image_path in glob.glob(cur_path + "/*.png"): 
                try:
                    img = image.load_img(image_path, target_size=image_size)
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    feature = model.predict(x)
                    flat = feature.flatten()
                    features.append(flat)
                    labels.append(label)
                    print ("[INFO] processed - " + str(count))
                    logger.debug("processed - %s" + str(count))
                    count += 1
                except Exception as e:
                    logger.error("ERROR:%s",e)
                    continue

            print ("[INFO] completed label - " + label)
            logger.debug("completed label -%s " + label)
    except Exception as e:
        logger.error("ERROR:%s",e)
    # encode the labels using LabelEncoder
    le = LabelEncoder()
    le_labels = le.fit_transform(labels)

    # get the shape of training labels
    #print ("[STATUS] training labels: {}".format(le_labels))
    print ("[STATUS] training labels shape: {}".format(le_labels.shape))
    #logger.debug("[STATUS] training labels:%s",le_labels)
    logger.debug("[STATUS] training labels shape:%s",le_labels.shape)

    # end time
    end = time.time()
    print ("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    logger.debug("[STATUS] end time - %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    end_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M") #datetime.datetime.now()
    total_time=end - start #end_time - start_time
    logger.debug("[STATUS] Total run time - %s", total_time)
    #features,le_labels
    features_shape=features[0].shape
    le_labels_shape=le_labels.shape
    logger.debug("features_shape:%s",features_shape)
    logger.debug("le_labels_shape:%s",le_labels_shape)
    
    response_json={"model_path":model_path,"train_path":train_path,"image_size":image_size,
                   "features":features, "le_labels":le_labels,
                   "features_shape":features_shape, "le_labels_shape":le_labels_shape,
                   "start_time":start_time,"end_time":end_time,"total_time(s)":total_time
                   }
    #response_json = json.dumps(response_json) #convert in json string obj
    return response_json
def save_features(model_path,model,features_path,features,labels_path,le_labels):
    # save features and labels
    logger.info("saving.. features and labels ")
    h5f_data = h5py.File(features_path, 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(features))

    h5f_label = h5py.File(labels_path, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(le_labels))

    h5f_data.close()
    h5f_label.close()

    # save model and weights
    model_json = model.to_json()
    with open(model_path  + ".json", "w") as json_file:
        json_file.write(model_json)

    # save weights
    model.save_weights(model_path  + ".h5")
    print("[STATUS] saved model and weights to disk..")
    logger.info("[STATUS] saved model and weights to disk..")

    print ("[STATUS] features and labels saved..")
    logger.info("[STATUS] features and labels saved..")
    features_shape=features[0].shape
    le_labels_shape=le_labels.shape
    logger.debug("features_shape:%s",features_shape)
    logger.debug("le_labels_shape:%s",le_labels_shape)
    response_json={"model_path":model_path,"features_path":features_path,"features_shape":features_shape,"le_labels_shape":le_labels_shape, "Status":"Completed and saved features and labels"}
    return response_json
