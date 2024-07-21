# Fire Detection Model using YOLO-V8 for raspberry pi

- This is a fire detection model using YOLO-V8 for raspberry pi. The model is trained on the custom dataset of fire images. The model is trained on
- **continuous_fire_dataset** dataset which contains 1000 images of fire and 1000 images of non-fire. The model is trained on the darknet framework.

## How to setup the model

- clone the repository using the following command:
  git clone https://github.com/jagjeet-singh-23/fire_detection.git
- install the required libraries using the following command:
  pip install -r requirements.txt

## How to run the model

- run the following command inside the terminal:
  cd fire_detection
  python ultralytics/yolo/v8/detect.predict.py model="best.pt" source=0 show=True
