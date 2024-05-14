"""[summary]
 @author Duy Nguyen Ngoc - duynguyenngoc@hotmail.com
"""

import json
import logging
import cv2
from celery import Celery, Task
from init_broker import is_broker_running
from init_redis import is_backend_running

from settings import config
from mq_main import redis


from worker.ml.model import CompletedModel
from worker.ml.helpers import image_utils as ml_image_helper
from helpers import time as time_helper
from settings import (celery_config, config, ml_config)
from helpers.storage import create_path


if not is_backend_running(): exit()
if not is_broker_running(): exit()


app = Celery(celery_config.QUERY_NAME, broker=config.BROKER, backend=config.REDIS_BACKEND)
app.config_from_object('settings.celery_config')


class PredictTask(Task):
    """
    Abstraction of Celery's Task class to support loading ML model.
    """
    abstract = True

    def __init__(self):
        super().__init__()
        self.model = None

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e. first task processed)
        Avoids the need to load model on each task request
        """
        if not self.model:
            logging.info('Loading Model...')
            self.model = CompletedModel()
            logging.info('Model loaded')
        return self.run(*args, **kwargs)
    
@app.task(bind=True, 
          base=PredictTask,
          name="{query}.{task_name}".format(
              query=celery_config.QUERY_NAME, 
              task_name=celery_config.ML_OBJECT_DETECTION_TASK_NAME))
def object_detection_task(self, task_id: str, data: bytes):
    """_summary_: object_detection by efi d2 model
    Args:
        task_id (str): _description_
        data (bytes): _description_

    Returns:
        _type_: _description_
    """

    data = json.loads(data) # load session data
    time = time_helper.now_utc()
    data['time']['start_detection'] = str(time_helper.now_utc().timestamp())
    string_time = time_helper.str_yyyy_mm_dd(time)
    try:
        image = ml_image_helper.read_image_from_path_to_numpy(data['upload_result']['path'])
        result = self.model.retrieval(image, data['upload_result']['path'])
        data['result'] = result
        data['time']['end_detection'] = str(time_helper.now_utc().timestamp())
        data['status']['detection_status'] = "SUCCESS"
        data['status']['general_status'] = "SUCCESS"
        data_dump = json.dumps(data)
        redis.set(task_id, data_dump) 
    except Exception as e:
        data['time']['end_detection'] = str(time_helper.now_utc().timestamp())
        data['status']['detection_status'] = "FAILED"
        data['status']['general_status'] = "FAILED"
        data['error'] = str(e)
        data_dump = json.dumps(data)
        redis.set(task_id, data_dump)
        