# Install Dependecy
pip install ultralytics


# Create data.yaml
echo -e "names:\n\
- fire\n\
- default\n\
- smoke\n\
nc: 3\n\
roboflow:\n\
  license: CC BY 4.0\n\
  project: fire-wrpgm\n\
  url: https://universe.roboflow.com/custom-thxhn/fire-wrpgm/dataset/8\n\
  version: 8\n\
  workspace: custom-thxhn\n\
test: ../data/D-Fire-datasets/fire/test/images\n\
train: ../data/D-Fire-datasets/fire/train/images\n\
val: ../data/D-Fire-datasets/fire/valid/images" > ../data/data.yaml


yolo task=detect mode=train model=yolov8s.pt data= dfire.yaml epochs=25 imgsz=416 plots=True

# Finished
echo "Model trained! Check ../data/D-Fire-datasets/runs/train3/weight/best.pt"