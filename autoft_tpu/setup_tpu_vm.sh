#!/bin/bash

fusermount -u ~/robust-ft
gcsfuse --implicit-dirs robust-ft ~/robust-ft
ls ~/robust-ft
source /home/carolinechoi/robust-optimizer/ropt/bin/activate
export PYTHONPATH="${PYTHONPATH}:/home/carolinechoi/robust-optimizer/autoft_tpu/"
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
