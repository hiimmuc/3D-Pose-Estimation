pip install -U openmim
mim install mmengine
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
mim install "mmdet>=3.1.0"

echo -e "Installing mmpose..."
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .

echo -e "Installing mmdetection..."
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .