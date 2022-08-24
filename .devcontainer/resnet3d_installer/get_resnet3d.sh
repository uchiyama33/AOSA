pip uninstall resnet3d -y
rm -rf resnet3d
git clone https://github.com/kenshohara/3D-ResNets-PyTorch.git
mv 3D-ResNets-PyTorch resnet3d
cp resnet3d_installer/resnet3d_setup.py resnet3d/setup.py
bash -c "python3.8 resnet3d_installer/modify_org3dressource.py"
cp resnet3d/utils.py resnet3d/models/utils.py

cp resnet3d_installer//main_init.py resnet3d/__init__.py
cd resnet3d
bash -c "python3.8 setup.py install"
cd ..

bash -c "python3.8 -c \"from resnet3d import generate_model\" "
