# Vision to Movement Program
Pathing program to translate image to set path, avoiding obstacles

Package installation:
```
pip install opencv-python numpy matplotlib scipy
```

### Return files
#### full_path.csv
waypoints, start and end points
#### waypoints.csv
only waypoints
#### occGrid.csv
Occupation grid boolean map
#### commands.csv
Start, end, distance and angle between each point

## main.py
- Defines parameters
- Writes CSV files

## vision.py
takes image and applies filters to isolate the obstacles

## planner.py
runs annealing algorithm and plots points and returns map

## planner.py
takes best path and converts it to set of instructions