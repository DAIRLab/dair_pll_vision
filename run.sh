#!/bin/sh

# 00: new data new mask

### iter 0 (tagslam trajectory)
# python examples/contactnets_vision.py  --run-name=00 --vision-asset=cube_1 --cycle-iteration=0 --clear-data #--bundlesdf-id=B
# python examples/contactnets_vision.py  --run-name=00 --vision-asset=cube_2 --cycle-iteration=0 --clear-data 
# python examples/contactnets_vision.py  --run-name=00 --vision-asset=cube_3 --cycle-iteration=0 --clear-data 
# python examples/contactnets_vision.py  --run-name=00_monitor_train_trajerror --vision-asset=cube_4 --cycle-iteration=0 #--clear-data 


### iter 1 (bundlesdf trajectory)
# python examples/contactnets_vision.py  --run-name=01 --vision-asset=cube_1 --cycle-iteration=1 --bundlesdf-id=00 --clear-data
python examples/contactnets_vision.py  --run-name=01 --vision-asset=cube_2 --cycle-iteration=1 --bundlesdf-id=00
# python examples/contactnets_vision.py  --run-name=01 --vision-asset=cube_3 --cycle-iteration=1 --bundlesdf-id=00
# python examples/contactnets_vision.py  --run-name=01 --vision-asset=cube_4 --cycle-iteration=1 --bundlesdf-id=00

# ### change output without rerunning pll (need to reconfigure the file)
# python bundlesdf_interface.py