#!/bin/sh

# Check for environment variable and set command accordingly
if [[ -n ${index_pos} ]]; then
  COMMAND="python inference.py -efol -pfol /home/seatizen/plancha -jgpu -mlgpu -c -ip $index_pos"
elif [[ -n ${index_start} ]]; then
  COMMAND="python inference.py -efol -pfol /home/seatizen/plancha -jgpu -mlgpu -c -is $index_start"
else
  COMMAND="python inference.py -efol -pfol /home/seatizen/plancha -jgpu -mlgpu -c"
fi

# Execute the command
exec $COMMAND