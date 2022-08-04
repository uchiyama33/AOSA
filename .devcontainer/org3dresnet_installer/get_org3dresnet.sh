pip uninstall org3dresnet -y
rm -rf org3dresnet
git clone https://github.com/kenshohara/3D-ResNets-PyTorch.git
mv 3D-ResNets-PyTorch org3dresnet
cp org3dresnet_installer/org3dresnet_setup.py org3dresnet/setup.py
bash -c "python3.8 org3dresnet_installer/modify_org3dressource.py"

cp org3dresnet/utils.py org3dresnet/models/utils.py

cp org3dresnet_installer//main_init.py org3dresnet/__init__.py
cd org3dresnet
bash -c "python3.8 setup.py install"
cd ..

bash -c "python3.8 -c \"from org3dresnet import generate_model\" "
