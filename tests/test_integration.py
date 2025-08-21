from __future__ import annotations

import ee

from downclim.downclim import define_DownClimContext_from_file
from downclim.logging_config import setup_logging

logger = setup_logging(level="INFO")

ee.Initialize(project = "downclim")

DownClimContext_example = define_DownClimContext_from_file("./DownClimContext_example.yaml")
DownClimContext_example.download_data()
DownClimContext_example.run_downscaling()
DownClimContext_example.run_downscaling(downscaling_grid_file="./results/chirps/chirps_Vanuatu_grid.nc")
DownClimContext_example.run_evaluation()
