# devreact
Linear ballistic accumulator model of reaction time through development.

## Installation

First, clone the project on your machine using git. 
Install [pyenv](https://github.com/pyenv/pyenv), then use it to install and activate python 3.10.
Finally, install [Poetry](https://python-poetry.org/).

To install the package and its dependencies using poetry:

```bash
cd [cloned project main directory]
poetry install
```

To use the installed package with Jupyter Lab:

```bash
python -m ipykernel install --user --name devreact
```

Next, edit the `.profile` to set the location to data and figures directories on your system.
Then launch Jupyter Lab:

```bash
. .profile
jupyter lab & 
```

You should be able to select the devreact kernel when opening an existing notebook or creating a new one.
