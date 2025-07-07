import yaml
import numpy as np
import rowan as rn
import json

from scipy.optimize import minimize


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
        R = np.array(flat).reshape(3, 3)
        print(f"R shape: {R.shape}, R: {R}")
        #R = np.eye(3)
        quat = rn.from_matrix(R)
        # change order to [w, x, y, z]
        quat = reorder_quat(quat)
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



#DERIVED FROM: https://github.com/whoenig/uav_trajectories/blob/main/scripts/generate_trajectory.py
def generate_straight_line_trajectory(
    start,
    end,
    start_vel:   np.ndarray = np.array([0.0, 0.0, 0.0]),
    end_vel:     np.ndarray = np.array([0.0, 0.0, 0.0]),
    max_vel:     float      = 1.5,
    max_acc:     float      = 2.0,
    max_snap:    float      = 10.0,
    dt:          float      = 0.01
) -> np.ndarray:
    """
    Generate a straight-line trajectory from start to end using
    a minimum-snap 7th-order polynomial in the path distance,
    with SciPy optimization to find the minimal feasible duration T.

    Returns:
        traj: (N,13) array [t, px, py, pz, vx, vy, vz, ax, ay, az, jx, jy, jz]
    """
    start = np.asarray(start, float)
    end   = np.asarray(end,   float)
    delta = end - start
    L = np.linalg.norm(delta)
    if L < 1e-8:
        return np.zeros((0,13))

    direction = delta / L
    v0 = np.dot(start_vel, direction)
    v1 = np.dot(end_vel,   direction)

    def poly7_coeffs(T):
        # boundary conditions at t=0
        a0, a1, a2, a3 = 0.0, v0, 0.0, 0.0

        # compute powers of T
        T2 = T * T
        T3 = T2 * T
        T4 = T2 * T2
        T5 = T3 * T2
        T6 = T3 * T3
        T7 = T4 * T3

        # system to solve for a4..a7 at t=T
        A = np.array([
            [   T4,    T5,    T6,     T7],
            [ 4*T3,  5*T4,  6*T5,   7*T6],
            [12*T2, 20*T3, 30*T4,  42*T5],
            [ 24*T,  60*T2,120*T3, 210*T4]
        ])
        b = np.array([
            L - (a0 + a1*T),
            v1 - a1,
            0.0,
            0.0
        ])
        a4, a5, a6, a7 = np.linalg.solve(A, b)
        return np.array([a0, a1, a2, a3, a4, a5, a6, a7])

    # sample profiles for constraint checking
    Ncon = 100
    def sample_profiles(T):
        coeffs = poly7_coeffs(T)[::-1]  # highest→lowest for np.polyval
        t_s = np.linspace(0, T, Ncon)
        s   = np.polyval(coeffs, t_s)
        v   = np.polyval(np.polyder(coeffs, 1), t_s)
        acc = np.polyval(np.polyder(coeffs, 2), t_s)
        j   = np.polyval(np.polyder(coeffs, 3), t_s)
        return v, acc, j

    # constraints: max|v| <= max_vel, etc.
    def c_vel(x):
        v, _, _ = sample_profiles(x[0])
        return max_vel - np.max(np.abs(v))

    def c_acc(x):
        _, acc, _ = sample_profiles(x[0])
        return max_acc - np.max(np.abs(acc))

    def c_snap(x):
        _, _, j = sample_profiles(x[0])
        return max_snap - np.max(np.abs(j))

    cons = [
        {'type': 'ineq', 'fun': c_vel},
        {'type': 'ineq', 'fun': c_acc},
        {'type': 'ineq', 'fun': c_snap},
    ]

    # initial guess for T: distance / max_vel
    T0 = max(L / max_vel, 1e-3)
    res = minimize(lambda x: x[0],
                   x0=[T0],
                   bounds=[(1e-6, None)],
                   constraints=cons,
                   options={'ftol':1e-6})

    if not res.success:
        raise RuntimeError(f"Optimizer failed: {res.message}")

    T_opt = res.x[0]

    # final dense sampling at the user’s dt
    t_arr = np.arange(0, T_opt + dt/2, dt)
    coeffs = poly7_coeffs(T_opt)[::-1]
    s   = np.polyval(coeffs, t_arr)
    v   = np.polyval(np.polyder(coeffs, 1), t_arr)
    acc = np.polyval(np.polyder(coeffs, 2), t_arr)
    j   = np.polyval(np.polyder(coeffs, 3), t_arr)

    traj = np.zeros((len(t_arr), 13))
    traj[:, 0]      = t_arr
    traj[:, 1:4]    = start + np.outer(s,   direction)
    traj[:, 4:7]    = np.outer(v,   direction)
    traj[:, 7:10]   = np.outer(acc, direction)
    traj[:, 10:13]  = np.outer(j,   direction)

    return traj


# print(load_start_goal_states('../../data/run_data.json'))
