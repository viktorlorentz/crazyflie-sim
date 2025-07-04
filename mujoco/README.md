# Crazyflie Simulation with MuJoCo

This repository contains code to simulate the Crazyflie quadcopter using MuJoCo. Follow the instructions below to set up and run the simulation.

## Installation

1. Clone the repository:
```sh
    git clone https://github.com/yourusername/crazyflie-sim.git
    git submodule sync
    git submodule update --init --recursive
```
2. Build the python bindings of the crazyflie-firmware. When you do `make -j4`, it might fail but it doesn't matter since we care about the python bindings and this will work when you do `make bindings_python` . DO not forget to set your pythonpath.
```sh
    cd path/to/crazyflie-firmware
    make cf2_defconfig
    make -j4
    make bindings_python
    export PYTHONPATH=path/to/crazyflie-sim/deps/crazyflie-firmware:$PYTHONPATH
```
## Usage

- To run the simulation for a single crazyflie, execute the following command:
```sh
    cd crazyflie-sim/mujoco/scripts/
   python3 model.py --traj_path ../../data/1cf_figure8_traj_fast_withoutpayload.csv --models_path ../models/dynobench/cf.yaml --mj ../models/xml/crazyflie.xml -v
```
- To run the simulation for a `n` crazyflies with payloads WITHOUT tendons (i.e., with rigid links), execute the following command:
```sh
    cd crazyflie-sim/mujoco/scripts/
python3 model.py --traj_path ../../data/2_robots_payload.yaml --models_path ../models/dynobench/2payload.yaml -p --mj ../models/xml/2cfs_payload.xml -v
```
- To run the simulation for a `n` crazyflies with payloads WITH tendons (i.e., without rigid links), execute the following command:
```sh
    cd crazyflie-sim/mujoco/scripts/
python3 model.py --traj_path ../../data/2_robots_payload.yaml --models_path ../models/dynobench/2payload.yaml -p -t --mj ../models/xml/2cfs_payload_tendons.xml -v
```

- To run the simulation and track a goal pose only for the payload and log the results to compare them: 
 ```sh
 python3 log_data.py 
 ```
 Inside you will find the commands necessary to check each run by itself.
### Notes
- You need `-p` to run the payload controller, and `-t -p` for payload models with tendons (because it has a different state vector). You also need `-v` to open the visualizer.
- Make sure that the lengths of the tendons in the `.xml` file are the same as your `run_data.json` for the `cable_length`. Similarly for the masses of the cfs and the payload.