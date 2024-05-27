# ShopperSim
A simulation tool to simulate customer behavior in retail store using Sorenson 2017 et. al. parameter
## Installation
Bigsimr is a library used in Julia. You cannot directly use it in Python. ref: https://pypi.org/project/bigsimr/
You have to install Julia and specify Julia's Path in Terminal first unless otherwises will error. 
To specify path, open your terminal and type: 
```sh
export PATH="/Applications/Julia 1.4.2.app/Contents/Resources/julia/bin:$PATH"
```
and
```sh
sudo ln -s /Applications/Julia-1.4.2.app/Contents/Resources/julia/bin/julia /usr/local/bin/julia
```
Note: This depends on your Julia version

ref: https://stackoverflow.com/questions/62905587/julia-command-not-found-even-after-adding-to-path

Also, libraries needed to be installed

```sh
pip install simpy
pip install git+https://github.com/SchisslerGroup/python-bigsimr.git
pip install ipywidgets
pip install ipython
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```
## Usage
Usage with Python IDE, i.e. Jupyter Notebook
```
from shopper_sim import create_widget
create_widget()

```




