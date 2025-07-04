import yaml
import numpy as np
import rowan as rn
import json


def saveyaml(file_out, data):
    with open(file_out, "w") as f:
        yaml.safe_dump(data,f,default_flow_style=None)

def loadyaml(file_in):
    with open(file_in, "r") as f: 
        file_out = yaml.safe_load(f)
    return file_out


def loadcsv(filename):
    return np.loadtxt(filename, delimiter=",", skiprows=1, ndmin=2)


def derivative(vec, dt):
    dvec = []
    # dvec  =[[0,0,0]]
    for i in range(len(vec) - 1):
        dvectmp = (vec[i + 1] - vec[i]) / dt
        dvec.append(dvectmp.tolist())
    dvec.append([0, 0, 0])
    return np.asarray(dvec)


def reorder_quat(quat):
    """
    Reorder quaternion from [x, y, z, w] to [w, x, y, z]
    """
    quat_reordered =  np.array([quat[3], quat[0], quat[1], quat[2]])
    return quat_reordered



def load_json(filename):    
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def save_json(out_path, data):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_start_goal_states(filename: str):
    
    data = load_json(filename)
    # start time
    # payload
    payload_start = data['payload_pos'][0]      # [x, y, z]
    # payload_linvel is stored per-quad but identical for each, so take quad 0
    payload_vel   = data['payload_linvel'][0][0]  # [vx, vy, vz]
    # target position
    target_pos = data['target_pos']


    # quads
    quad_starts = data['quad_pos'][0]           # [[x1,y1,z1], [x2,y2,z2], ...]
    rot_flats   = data['agent_rot_flat'][0]     # [[r11..r13,r21..r23,r31..r33], ...]
    rots = np.array(rot_flats).reshape((-1, 3, 3))  # reshape to (runs, quads, 3, 3)
    linvels     = data['agent_linvel'][0]       # [[vx,vy,vz], ...]
    angvels     = data['agent_angvel'][0]       # [[vx,vy,vz], ...]
   
    start_qpos= []
    start_qpos.extend(payload_start)
    start_qpos.extend([1.0,0.0,0.0,0.0]) # payload quat dummy
    start_qvel= []
    start_qvel.extend(payload_vel)
    start_qvel.extend([0.0,0.0,0.0]) # payload ang vel dummy


    quat_agents = []
    for i, (pos, flat, linvel, angvel) in enumerate(zip(quad_starts, rot_flats, linvels, angvels)):
        start_qpos.extend(pos)        
        R = np.array(flat).reshape((3, 3))
        # R = np.eye(3)
        quat = rn.from_matrix(R)
        start_qpos.extend(quat) 
        start_qvel.extend(linvel)
        start_qvel.extend(angvel)

    start_st = []
    start_st.extend(start_qpos)
    start_st.extend(start_qvel)

    traj_data = dict()
    traj_data["start_state"] = start_st
    traj_data["trajectory"] = data["trajectory"] 
    if data["trajectory"] is not None:
        traj_data["target_pos"] = data["trajectory"]
    else:
        traj_data["target_pos"] = target_pos
    traj_data["dt"] = data["dt"]
    traj_data["l_payload"] = data["env_config"]["cable_length"]
    traj_data["m_payload"] = data["env_config"]["payload_mass"]
    return traj_data



# print(load_start_goal_states('../../data/run_data.json'))
