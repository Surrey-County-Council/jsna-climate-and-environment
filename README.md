# Useage

the following modules all have a function named get_data which accepts an output path and a method for overwriting cached data

this specification can be used to run the end to end process of getting data required or reading the data locally.

local testing within each of the scripts automatically overwrites the data.

# Setup

dependencies for the project are stored in the pyproject.toml and the easiest way to run the code is using uv:

```powershell
.../jsna-climate-and-environment> pip install uv
.../jsna-climate-and-environment> uv sync
.../jsna-climate-and-environment> uv run main
```

for additional development and experimentation feel free to use the modules provided for deeper analysis.
dev dependencies include jupyter notebooks to facilitate experimentation

# Modules

geography - connects to 2 different arcgis api endpoints. useful for downloading nspl, boundary files and met office datasets. see function doccumentation if you wish to use these for analysis.

datasets - each module contains transformation functions for a single dataset. see function doccumentation if you wish to use these for analysis.

# Environment Variables
OUTPUT_DIR - can be set to output data to a specific location. By default (if unset) data will be written to `.../jsna-climate-and-environment/data/output`