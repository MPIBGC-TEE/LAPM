# although the following packages are mentioned in setup.py of bgc_md as dependencies
# the install in the order determined by setuptools seems to fail 

pip3 install --upgrade pip
pip3 install numpy
pip3 install sympy
pip3 install matplotlib
pip3 install scipy
pip3 install concurrencytest

# $1 could be 'develop'
python3 setup.py $1
