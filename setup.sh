git clone https://github.com/devaansh100/LightXML.git
cd LightXML
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install
cd ..
pip install transformers
tar -xvzf ../drive/MyDrive/Wiki-500K.tar.gz # Change path
mkdir data/Wiki-500K
mv Wiki-500K/* data/Wiki-500K
rmdir Wiki-500K
./run.sh wiki500k