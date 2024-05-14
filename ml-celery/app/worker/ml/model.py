import numpy as np
import tensorflow.compat.v2 as tf
from worker.ml.helpers import load_label_map
from settings import ml_config
from mtcnn import MTCNN
from PIL import Image
import numpy as np
from keras.models import load_model
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from numpy import loadtxt
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
import pyspark.sql.functions as f
import os 

class FaceExtraction(object):
    def __init__(
        self, 
        path_to_model
    ):
        self.model = load_model(path_to_model)
        self.detector = MTCNN()

        spark = SparkSession.builder.appName("LSHS1").getOrCreate()
        vectors = os.listdir(ml_config.VECTOR_PATH)
        data = []
        for vector in vectors:
            array = loadtxt(os.path.join(ml_config.VECTOR_PATH, vector))
            data.append((vector, Vectors.dense(array)))
        self.df = spark.createDataFrame(data, ["id", "features"])
        self.lsh_model = BucketedRandomProjectionLSHModel.load(ml_config.LSH_MODEL_PATH)


    def extract_face(self, img, path, required_size=(150, 150)):
        results = self.detector.detect_faces(img)

        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        face = img[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        image.save(path.replace("upload", "object-detection"))
        return path.replace("upload", "object-detection")


    def predict(self, image, path):
        path_extracted = self.extract_face(img=image, path=path)
        single_img_df = pd.DataFrame({'file_path': [path_extracted], 'label': ['detector']})
        predict_datagen = ImageDataGenerator(rescale=1./255) 
        single_tensor = predict_datagen.flow_from_dataframe(single_img_df, x_col='file_path', y_col='label',
                                                target_size=(224, 224),
                                                batch_size=32, class_mode='categorical')
        
        img = single_tensor[0][0]
        test_prediction = self.model.predict(img, verbose=0)
        return test_prediction[0]



    def retrieval(self, image, path):
        key = Vectors.dense(self.predict(image, path))

        result = self.lsh_model.approxNearestNeighbors(self.df, key, 10)
        list_id = result.select(f.collect_list('id')).first()[0]
        list_link = []
        list_name = []
        origin_path = "https://storage.googleapis.com/face_retrieval/dataset"
        for id in list_id:
            image_name = id.split("_")[-1].replace(".npy","")
            name = "_".join(id.split("_")[:0-1])
            list_link.append(f"{origin_path}/{name}/{image_name}")
            list_name.append(" ".join(id.split("_")[:0-1]))
        return {"link": list_link, "name": list_name}



class CompletedModel(object):
    def __init__(self):
        self.model = self._load_model()
        
    @staticmethod
    def _load_model():
        return DetectorTF2(
            path_to_model=ml_config.MODEL_PATH,
            path_to_labels=ml_config.LABLE_PATH,
            nms_threshold=ml_config.NMS_THRESHOLD, 
            score_threshold=ml_config.SCORE_THRESHOLD,
            num_classes=ml_config.NUMBER_CLASSES,
            max_classes_out=ml_config.MAX_CLASS_OUT
        )
    
    def retrieval(self, image, path):
        return self.model.retrieval(image, path)
