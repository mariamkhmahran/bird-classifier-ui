# Bird Species Classification Web Application

This repository contains a Flask web application for the classification of bird species. The system utilizes a pre-trained TensorFlow 2 model for bird detection, species identification, and localization. The application consists of three main parts: the web interface built using the Flask framework, the model served using TFServing, and the database configured using MySQL.

## Objective

The main goal of the system is to provide a scalable object detection system with a user-friendly web interface. The system specifically focuses on bird species detection, allowing users to upload images in various formats. The system should be able to run inference on the uploaded media and display the predictions on a results page. Users should also be able to provide feedback on the system's results.

## Getting Started

To run the bird species classification web application, follow these steps:

1. Install Docker and Docker Compose on your machine.
2. Clone this repository to your local system.
3. Navigate to the repository directory.
4. Make sure that MySQL is running on the localhost.
5. From the root folder, run the following command: `mysql -u <username> -p < ./db/init.sql` to initialize the database.
6. Go to `./client` subfolder, open app.py and update the MySQL config object with relevant configuration to your local MySQL. (the config object is defined right after import statements).
7. From the root directory, build the Docker image: `docker build -t bird-classification-app .`
8. Start the application using Docker Compose: `docker-compose up`
9. Access the web application in your browser at `http://localhost:8080`

## System Design

The system follows MLOps best practices and incorporates various technologies to ensure efficient deployment and management of the bird detection model. The workflow pipeline includes model serving, data storage, web application hosting, and monitoring. The system design utilizes containerization techniques for scalability and reproducibility.

The system design is based on the level 1 MLOps maturity level as outlined by Google. Here is an overview of the different components and their functionalities:

### Model Overview

The object detection model used in this project is a Faster R-CNN Resnet-101 V1 network. The model detects the presence of birds in an input image, classifies them into European robin, Coal Tit, and Eurasian magpie species, and localizes the birds within the image. Transfer learning was applied from the TensorFlow model zoo, and the model was further refined using a dataset of 3000 bird pictures. The model achieved an accuracy of 91% in testing, with mAP@0.50IOU of 0.863 and mAP@0.75IOU of 0.771. The model can be used for bird population monitoring, migration tracking, and studying bird behaviors in their natural habitats. After training and testing the model, it was frozen in preparation for deployment. The model was exported in the Google .pb file format which is directly used in this repository.

### Model Serving

The frozen model is served using TFServing, a flexible interface for serving models in production environments. TFServing supports multiple protocols, such as REST API and gRPC, and provides model version management. In this setup, the served model is a frozen model stored in the `./server/saved_model` directory. The server itself operates within a Docker container, using the `tensorflow/serving` image obtained from Docker Hub.

The server actively listens to the REST API port 8501, which is exposed for communication. When an image is submitted via a POST request to this endpoint, the server promptly responds with the prediction results pertaining to the given image. These prediction results encompass information like the detected classes, corresponding confidence scores, and the bounding boxes indicating the location of the detected objects.

### Flask Web Interface

A user-friendly web interface built using Flask to facilitate interaction with the model. The root directory of the is in `./client`. The app is developed using python, css and html. Bootstrap and JQuery are used for styling and JS operations. Both libraries are included in the project using CDN links.

The app consists of three pages:

1.  **Home page:**

This page allows the user to upload any image in png, jpg or jpeg format. When the user click on upload button, the image is saved locally to the `./client/originals` folder. The image is also stored into the `images` table in the database with a unique ID. The page then redirects to `uploads/<ID>` where ID maps to the unique image ID in the images table inside the database.

2. **Results page:**

This page displays the inferencing results of an image. The original image is displayed as well as the inferenced image for comaprison. The name and unique ID of the image are also visible on the page. Additionally, each prediction is listed in a table where each row contains the name of the predicted class and the confidence score.

The url of this page is formatted as `uploads/<ID>`. This ID maps to the unique ID of this image in the images table in the database. When the page is first loaded, the page first looks for a record with the same ID in the inferences table.

- If such a record exists then this image was already inferenced and saved in the database. In this case, the previous results are displayed immediately.
- Otherwise, the _inference_ function is called to send the image over to the server and get back the results. The new results are then stored in the database for future use. The resulting image with the bounding boxes is saved locally to the `./client/inferenced` folder.

This page also contains a feedback section. This section allows the user to provide feedback on the quality of the results. This feedback helps in monitoring the performance of the model overtime to detect if a feature drift is happening or if the model requires retraining.

The section contains a form asking the user whether or not the model had captured all existing birds in an image.

- If the user votes yes, then the user satisfaction rate for this image is considered 100%.

- If the the user votes no, then the user is asked to enter the number of birds present in the picture (ground truth) and the number of birds that were _accurately_ detected (predicted value). The user satisfaction rate is then calculated as predicted_value/ground_truth. (e.g. if the model predicts one of two birds correctly, then the user satisfaction rate is 50%).

The results of this form are saved into the `monitoring` table in the database.

3. **Monitoring page:**

This page displays all the data stored in the `monitoring` table mentioned above. The page lists basic data about the model and the number of images uploaded to the app since its launch. The page also contains a graph illustrating and comparing the trends of the average scores of the 3 classes and the user satisfaction rate over time.

This page is extremely beneficial for the monitoring stage of the workflow. It serves as an indicator for the reliability of this model and whether it requires re-training or fine tuning.

### Data Storage

An external MySQL database is used to store various elements across the pipeline, including user-uploaded pictures, prediction results, user feedback, and analytical values such as average confidence scores. MySQL is chosen for its reliability and widespread usage, providing efficient management, storage, and retrieval of data.

The database used is called `birdsClassifier` and it contains three main tables:

- images: This table stores the image blob and image name under a unique ID. Each row also contains a boolean value indicating whether or not the user has voted for the quality of inferencing for this image.
- inferences: This table stores the inferencing results for images. Each record in the table represents one boundin box. Each image may be linked to multiple inferencing records (bounding box). Each record contains the unique ID of the image, the coordinates of the bounding box, the confidence score for this prediction, and the class ID.
- monitoring: This table stores the results of user feedback and keeps track of the average confidence scores for each class overtime. Each row in this table contains the following data:

  - **imageID** of the id of the image related to those results.
  - **Timestamp** of when this feedback was submitted.
  - **Average confidence score for class 1** which is equal to the average score for this class over time.
  - **Average confidence score for class 2** which is equal to the average score for this class over time.
  - **Average confidence score for class 3** which is equal to the average score for this class over time.
  - **Average user staisfaction rate** over time.

  (with each feedback submitted those values are caclulated as the average between the score/rate in the last entry over the average score/rate in the new entry)

In addition to those 3 tables, the database typically includes 15 more tables that act as a metadate store. The tables are added to the databse from MLMD.

### Metadata Store

ML Metadata (MLMD) is used to manage model metadata and track model versions and experiments. MLMD keeps a record of how different versions of the model were trained and used, facilitating reproducibility and providing insights into model behavior and performance over time.

The metadata store is initialized by running `python mlmd.init.sql` in the root directory. This executes a script to connect to MySQL, initialize the store, and add the saved model to the store as an artifact.

### Containerization

Containerization plays a crucial role in this project by packaging the model and its dependencies into containers. This approach offers several benefits, including improved scalability and reproducibility. Two main components of the system benefit from containerization: the model serving component and the Flask web application.

Using Docker Compose, the project manages multiple containers simultaneously, enabling seamless communication between them through REST API or Docker networking. This approach provides flexibility and scalability, allowing the system to handle increased workloads efficiently.

### Model Monitoring

Continuous monitoring of the model's performance is crucial in any MLOps pipeline for ensuring optimal results. Two measures are implemented to indicate the need for retraining:

1. User feedback: Users provide feedback on the model's detection accuracy for each uploaded file. If the overall user satisfaction rate falls below 90%, it indicates the need for retraining.

2. Average confidence score: During the testing phase, the average confidence score per class is calculated. The average confidence score is then computed for each predicted class from user uploads. If the calculated score falls below the original average confidence score, it indicates the need for retraining.

If either of these values falls below the threshold, this serves to indicate the need for retraining. The wrongly predicted photos are manually collected and retagged, followed by a new training cycle with the updated data.

## Repository Structure

The repository structure is as follows:

```
- client/
  - inferenced/
  - originals/
  - templates/
    - index.html
    - results.html
    - monitor.html
    - 404.html
  - static/
    - index.css
    - assets/
  - app.py
  - Dockerfile
  - requirements.txt
  - ...
- server/
  - saved_model
    - 001/
      - variables/
        - ...
      - saved_model.pb
  - Dockerfile
- db
  - init.sql
- docker-compose.yml
- mlmd.init.py
- README.md
```

- The `client/` directory contains the Flask web application code. The `templates/` subdirectory includes the HTML templates for the web interface, `index.html` for file upload, `results.html` for displaying inference results, `monitor.html` for monitoring page and `404.html` as a error fallback page. The `static/` subdirectory contains the CSS stylesheets for the web application as well as relevant assets.
- The `server/` directory holds the frozen model files, including `saved_model.pb` and the `variables/` directory.
- The `db/` directory holds schema for the app data storage db.
- `docker-compose.yml` is the configuration file for Docker Compose, defining the services and their dependencies.
- `README.md` provides an overview of the repository and its components.

## Conclusion

This repository contains a Flask web application for bird species classification. The system comprises a web interface built using the Flask framework, a model served using TFServing with an HTTP endpoint, and a MySQL database for data storage. The model is a Faster R-CNN Resnet-101 V1 network trained to detect and classify European robins, Coal Tits, and Eurasian magpies. The application allows users to upload images or videos to identify bird species and provides detection and localization results. Continuous monitoring and feedback collection enable tracking of model performance and indicate the need for retraining. The repository provides a comprehensive solution for bird species classification and can be further extended and customized as needed.
