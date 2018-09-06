# although the following packagesmined by setuptools seems to fail 
# sometimes (probably because the dependencies are not completely resolved
# the following order of istallation helps in some of these cases
# Especially numpy requires some (C or Fortran) libs to be installed
# so it might be a good idea to install the numpy package of your distribution
# which will trigger the libs to be installed
# you can then later install newer versions of numpy in a virtualenv which will
# find the system wide istalled libs ...
pip3 install    numpy
pip3 install    sympy
pip3 install   matplotlib
pip3 install   scipy

pip3 install --upgrade pip
pip3 install -r requirements.developer

python3 setup.py develop
