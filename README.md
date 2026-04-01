# devreact
Linear ballistic accumulator model of reaction time through development.

## Installation

First, clone the project on your machine using git. [Install uv](https://docs.astral.sh/uv/getting-started/installation/). 
To install Python and pinned dependencies, run `uv sync`.

To use the installed package with Jupyter Lab:

```bash
uv run ipython kernel install --user --env VIRTUAL_ENV "$(pwd)/.venv" --name=devreact
```

Next, edit the `.profile` to set the location to data and figures directories on your system.
Then launch Jupyter Lab:

```bash
. .profile
uv run jupyter lab & 
```

You should be able to select the devreact kernel when opening an existing notebook or creating a new one.
