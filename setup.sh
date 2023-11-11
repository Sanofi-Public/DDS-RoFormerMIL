# wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
# sh Miniconda3-py38_4.12.0-Linux-x86_64.sh -bu
# rm Miniconda3-py38_4.12.0-Linux-x86_64.sh

# apt-get install libxrender1 -y
# apt-get install libgl1 -y

# source /root/miniconda3/etc/profile.d/conda.sh
# source ~/.bashrc
# conda init

conda create -n mil python=3.10 -y
conda activate mil
conda install -c conda-forge openslide -y
pip install -U scikit-learn==1.2.1 
pip install tiatoolbox==1.3.3
pip install torch==2.0 torchvision --force 
pip install -r requirements.txt

pre-commit install

echo "PYTHONPATH=${PYTHONPATH}:${PWD}:${PWD}/CLAM" > .env
export PYTHONPATH=${PYTHONPATH}:${PWD}:${PWD}/CLAM
