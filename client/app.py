from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO

import mysql.connector
import requests
import numpy as np 
import datetime
import random
import base64
import os
import sys
import cv2
import json

# import prediction service functions from TF-Serving API
from utils import label_map_util
from utils import visualization_utils as viz_utils
from core.standard_fields import DetectionResultFields as dt_fields
# import utils for stub creation and prediction
from utils.stubs_utils import Stub

############################################################################################
# Configuration ############################################################################

# MySQL configuration. Update this with the relevant user name and password
config = {
    'user': 'root',
    'password': 'ROOT',
    'host': 'localhost',
    'port': '3306',
    'database': 'birdsClassifier'
}
connection = mysql.connector.connect(**config)

sys.path.append("..")

# Constant values used throughout the app
PATH_TO_LABELS = "./data/label_map.pbtxt"
MODEL_NAME = "birdsClassifier"
NUM_CLASSES = 3

# Labels map for class IDs
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# initialize flask app and flask configuration variables
app = Flask(__name__)

app.config['ORIGINALS_FOLDER'] = 'originals/'
app.config['INFERENCED_FOLDER'] = 'inferenced/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

############################################################################################
# helper functions #########################################################################

def allowed_file(filename):
    ''' checks if file format is allowed '''

    return '.' in filename and \
            filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def convert_to_b64(ndarray):
    '''convert numpy array to base 64 image
    Args:
    ndarray: np array image
    Returns:
    a string representing base64 representation of the image
    '''

    arr = np.array(ndarray, dtype=np.uint8)
    im = Image.fromarray(arr)
    im_file = BytesIO()
    im.save(im_file, format="PNG")
    im_bytes = im_file.getvalue()
    im_b64 = base64.b64encode(im_bytes).decode('ascii')
    return im_b64

def get_class_name(id):
    ''' map the class ID to class name '''

    _class = category_index[id]['name']
    words = [[_class[0]]]
 
    for c in _class[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)
 
    return ' '.join([''.join(word) for word in words])

def query_db(query, val=None):
    ''' Initiates a connection with db, executes the query, then closes the connection
    Args:
    query: MySQL query to be executed
    vals: values of the query
    '''

    cursor = connection.cursor()
    cursor.execute(query, val)
    results = cursor.fetchall()
    
    cursor.close()
    connection.commit()

    return results

############################################################################################
# Inferencing & feedback functions #########################################################

def inference(img, ID):
    ''' sends image to TFServing for inferencing
    Args:
    img: 2D list ofthe image
    ID: unique ID of the image in database
    Returns:
    frame: the infrenced image with bounding boxes
    '''

    # put input image into request body as a tensor
    data = {"instances": [{"input_tensor": img}]}

    # create a stub, then send data for prediction
    stub = Stub(host="127.0.0.1", port="8501", model_name=MODEL_NAME)
    result = stub.predict(data)

    image_np = np.array(img) # np representation of the img
    pred = result['predictions'][0] # prediction results for input image

    output_dict = {}
    output_dict['classes'] = np.squeeze(pred[dt_fields.detection_classes]).astype(np.uint8)
    output_dict['boxes'] = np.reshape(pred[dt_fields.detection_boxes], (-1, 4))
    output_dict['scores'] = np.squeeze(pred[dt_fields.detection_scores])

    # Only keep boxes and predictions with confidence score > 75%
    length = len(output_dict['boxes'])
    output_dict['boxes'] = np.array([list(output_dict['boxes'][idx]) for idx in range(length) if output_dict['scores'][idx] >= 0.75][:50])
    output_dict['classes'] = np.array([output_dict['classes'][idx] for idx in range(length) if output_dict['scores'][idx] >= 0.75][:50])
    output_dict['scores'] = np.array([output_dict['scores'][idx] for idx in range(length) if output_dict['scores'][idx] >= 0.75][:50])

    # Save prediction in db for future use
    length = len(output_dict['boxes'])

    # each bounding box is stored as a separate record in the inferencing table
    # all bounding boxes are linked to the image with the image ID
    for idx in range(length):
        _box = output_dict['boxes'][idx]
        _box = ", ".join(map(str, _box))
        _score = output_dict['scores'][idx]
        _class = output_dict['classes'][idx]

        query = "INSERT INTO inferences (ID, box, score, class) VALUES (%s, %s, %s, %s)"
        val = (ID, _box, _score, int(_class))
        query_db(query, val)

    # Draw the bounding boxes in the image
    frame = visualize(image_np, output_dict)

    return (frame, output_dict)

def visualize(image_np, output_dict):
    '''Draws the bounding boxes in the image'''

    frame = viz_utils.visualize_boxes_and_labels_on_image_array(image_np,
                    np.array(output_dict['boxes'], dtype=np.float),
                    np.array(output_dict['classes'], dtype=np.int),
                    np.array(output_dict['scores'], dtype=np.float),
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=50,
                    min_score_thresh=.75,
                    agnostic_mode=False)
                    
    return (frame)

def handle_feedback(data, meta_data):
    ''' calculates the average scores and user satisfaction rates then
        stores those values into the `monitoring` table
    Args:
    data: feedback form values
    meta_data: image metadata, including ID, image name , and bounding boxes details
    '''

    ans = data.get('vote') # answer to the yes/no question
    ground_truth = float(data.get('ground_truth') or 0) # number of birds in the image
    predicted = float(data.get('predicted') or 0) # number of birds accurately detected

    instances = query_db("SELECT * FROM monitoring ORDER BY timestamp DESC")
    n = len(instances)

    # initialize variables with 100%
    avg_score_1 = 1.0 # avg confidence score for class 1
    avg_score_2 = 1.0 # avg confidence score for class 1
    avg_score_3 = 1.0 # avg confidence score for class 1
    user_satisfaction = 1.0 # avg user satisfaction rate

    if len(instances) > 0:
        # average scores/rates so far
        avg_score_1 = instances[0][2]
        avg_score_2 = instances[0][3] 
        avg_score_3 = instances[0][4]
        user_satisfaction = instances[0][5]

    # take the average between the current class score and the average score so far
    for obj in meta_data['objects']:
        if obj['class']['id'] == 1:
            avg_score_1 = ((avg_score_1 * n) + (obj['score']/100)) / (n + 1)
        if obj['class']['id'] == 2:
            avg_score_2 =  ((avg_score_2 * n) + (obj['score']/100)) / (n + 1)
        if obj['class']['id'] == 3:
            avg_score_3 =  ((avg_score_3 * n) + (obj['score']/100)) / (n + 1)

    if ans == 'yes'or ground_truth == predicted:
        ratio = 1 # 100% satisfaction
    elif ground_truth == 0 and predicted > 0:
        ratio = 0 # 0% satisfaction
    else:
        ratio = predicted / max(ground_truth, predicted) # ratio of user satisfaction with the current results

    # average between current user satisfaction rate and the average rate so far
    user_satisfaction = (user_satisfaction * n) + ratio
    user_satisfaction = user_satisfaction / (n + 1)
    
    # store values in database
    query = "INSERT INTO monitoring (imageID, avg_score_1, avg_score_2, avg_score_3, user_satisfaction) VALUES (%s, %s, %s, %s, %s)"
    val = (meta_data['ID'], avg_score_1, avg_score_2, avg_score_3, user_satisfaction)
    query_db(query, val)

    # set the voted boolean true in the images database
    # this value indicates that the vote was taken for the current image,
    # no need to show the form again in the future
    query = "UPDATE images SET voted=1 WHERE ID=%s"
    val = (meta_data['ID'],)
    query_db(query, val)

############################################################################################
# Flask pages routes #######################################################################

@app.route ('/')
def index():
    ''' Home page '''
    return render_template('index.html')

@app.route('/upload', methods= ['POST'])
def upload():
    ''' Endpoint for uploading images '''
    
    ID = random.getrandbits(32) # random ID, unique for this image in db

    file = request.files['file']
    if file and allowed_file(file.filename):
        blob = file.read()
        
        # save file in db with a unique ID
        query = "INSERT INTO images (ID, name, image_blob) VALUES (%s, %s, %s)"
        val = (ID, file.filename, blob)
        query_db(query, val)

        # save original file on disk
        extension = file.filename.split('.').pop()
        fpath = os.path.join(app.config['ORIGINALS_FOLDER'], '{}.{}'.format(ID, extension))
        with open(fpath, 'wb') as f:
            f.write(blob)

    # redirect to uploads/<ID>
    return redirect(url_for('uploaded_file', ID=ID))

@app.route('/uploads/<ID>', methods= ['POST', 'GET'])
def uploaded_file(ID):
    ''' Results page and endpoint for feedback submission '''
    feedback_submitted = False

    # Get image details from db by ID
    img_response = query_db("SELECT * FROM images WHERE ID=%s LIMIT 1", (ID, ))
    # check for previous inferencing results for the same image
    inference_response = query_db("SELECT * FROM inferences WHERE ID=%s", (ID, ))

    if len(img_response) > 0:
        img_response = img_response[0]
        img = img_response[2] # image blob
        feedback_submitted = bool(img_response[3])
        
        img_np = np.fromstring(img, dtype='uint8')
        image_content = cv2.imdecode(img_np,1)
        image_content = cv2.cvtColor(image_content, cv2.COLOR_BGR2RGB)

        if len(inference_response) == 0:
            # if no previous inferencing results for the same image was found
            # send the image to the server for prediction
            frame, output_dict = inference(image_content.copy().tolist(), ID)
        else:
            # else, display the old results right away
            output_dict = {
                'boxes': [],
                'classes': [],
                'scores': [],
            }
            for detection in inference_response:
                output_dict['boxes'].append(detection[1].split(','))
                output_dict['scores'].append(detection[2])
                output_dict['classes'].append(detection[3])

            frame = visualize(image_content.copy(), output_dict)

        im = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)

        im_original = convert_to_b64(image_content) # original image
        im_inf = convert_to_b64(frame) # inferenced images

        # save the metadate for this image to be sent to the html file
        meta_data = {
            'ID': img_response[0],
            'img_name': img_response[1],
            'objects': [{
                'class': {
                    'id': output_dict['classes'][idx],
                    'name': get_class_name(output_dict['classes'][idx])
                },
                'score': round(output_dict['scores'][idx] * 100, 2),
                }
                for idx in range(len(output_dict['boxes']))]
        }
    
        # save the inferenced image frame on disk in the `inferenced` folder
        ID = img_response[0]
        extension = img_response[1].split('.').pop()
        path = os.path.join(app.config['INFERENCED_FOLDER'], '{}.{}'.format(ID, extension))

        if not os.path.isfile(path):
            _im = Image.fromarray(frame.astype(np.uint8))
            _im.save(path)

        # if this page was called on the submit of the feeback form
        if request.method == "POST":
            # handle the submission using the helper function 
            handle_feedback(request.form, meta_data)
            feedback_submitted = True
        
        # render results page
        return render_template('results.html', is_submitted=feedback_submitted, meta_data=meta_data, original=im_original, inferenced=im_inf)
    
    # if the image ID wasn't found in db -> 404 page
    return render_template('404.html')

@app.route('/monitor')
def monitor():
    ''' Monitoring page where details and statistics about the model are displayed '''

    images = len(query_db("SELECT ID FROM images")) # number of uploaded images so far

    # monitoring stats
    results = query_db("SELECT * FROM monitoring ORDER BY timestamp ASC")

    if len(results) > 8:
        # display only last 8 records. This is due to the available space on a page.
        results = results[len(results)-8 : ]

    ticks = [] # x axis ticks
    avg_scores_1 = []
    avg_scores_2 = []
    avg_scores_3 = []
    user_satisfaction = []
    for elem in results: 
        [_,timestamp, avg1, avg2, avg3, satisfaction] = elem
        ticks += [timestamp.strftime("%y %b %H:%M")]
        avg_scores_1 += [avg1]
        avg_scores_2 += [avg2]
        avg_scores_3 += [avg3]
        user_satisfaction += [satisfaction]
        

    return render_template('monitor.html', ticks=ticks, avg1=avg_scores_1, avg2=avg_scores_2,
                                avg3=avg_scores_3, user_satisfaction=user_satisfaction, images=images)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
