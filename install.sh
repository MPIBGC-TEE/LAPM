# although the following packages are mentioned in setup.py of bgc_md as dependencies
# the install in the order determined by setuptools seems to fail 

pip3 install --upgrade pip
# install common requirements for developers and normal users
pip3 install -r requirements.developer
# install additional requirements that a developer usually has in develop mode
pip3 install -r requirements.txt

# $1 could be 'develop'
python3 setup.py $1
