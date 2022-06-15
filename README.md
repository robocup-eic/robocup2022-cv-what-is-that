# robocup2022-cv-what-is-that

Computer Vision models for "What Is That" task in RoboCup@Home 2022 <br>
ref : https://www.augmentedstartups.com/YOLOR-You-Only-Learn-One-Representation-PyTorch

# Installation
1. create conda environment using python 3.9.6
2. Install pytorch <br>
`$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`
3. Install Cython <br>
`$ pip install -U cython`
4. Install pycocotool <br>
`$ pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"`
5. Download yolor config folder from https://drive.google.com/file/d/17JveNhnGDXN2ilw_lLpfy6XLt7nj-sjd/view <br>
6. Put all downloaded files to /object_detection_module/config
7. Install requirement.txt <br>
`pip install -r requirements.txt`
