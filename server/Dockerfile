FROM tensorflow/serving:latest

COPY ./saved_model /models/birdsClassifier

EXPOSE 8501

CMD ["--rest_api_port=8501", \
        "--model_name=birdsClassifier", \
        "--model_base_path=/models/birdsClassifier"]

