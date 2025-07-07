import mujoco as mj
import numpy as np
import time
from mujoco.glfw import glfw
import cffirmware as cff
from helper import *
import argparse 
import rowan as rn
import cvxpy as cp 
import rowan as rn
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)


class LeePayloadController():
    def __init__(self, model_params):
        """Initialize the Lee controller."""
        self.ctrlLeeP = cff.controllerLeePayload_t()
        cff.controllerLeePayloadInit(self.ctrlLeeP)
        self.ctrlLeeP.mass = model_params["m"][0]
        self.ctrlLeeP.mp = model_params["m_payload"]
        self.l =  model_params["l_payload"]
        arm_length = 0.046  # m
        arm = 0.707106781 * arm_length
        t2t = 0.006  # thrust-to-torque ratio
        self.num_robots = model_params["num_robots"]
        self.team_ids = [i for i in range(self.num_robots)]
        self.ctrlLeeP.en_qdidot = 1
        self.ctrlLeeP.gen_hp = 1
        self.ctrlLeeP.en_accrb = 0
        if model_params["plan_type"] == "payload_target_pos":
            self.ctrlLeeP.formation_control = 1 # set this to 1 if you don't want to follow a formation 
        else: 
            self.ctrlLeeP.formation_control = 3 # set this to 1 if you don't want to follow a formation 
            
        self.gains = [
            (18, 12, 0),
            (18, 12, 0),
            (0.03, 0.0012, 0.0),
            (100, 100, 100),
            (100),
        ]
        self.state = cff.state_t()
        self.sensors = cff.sensorData_t()
        self.control = cff.control_t()
        self.setGains()
        self.ctrlLeeP.radius = model_params["col_size_robot"]
        self.state.num_uavs = self.num_robots
        self.setSetpoint()
        self.u_nominal = self.ctrlLeeP.mass * 9.81 / 4
        self.B0 = np.array([
            [1, 1, 1, 1],
            [-arm, -arm, arm, arm],
            [-arm, arm, arm, -arm],
            [-t2t, t2t, -t2t, t2t]
            ]) # allocation matrix that maps motor forces to collective thrust and torques
        self.B0_inv = np.linalg.inv(self.B0)
    
    def setGains(self):
        kpos_p, kpos_d, kpos_i = self.gains[0]
        kc_p, kc_d, kc_i = self.gains[1]
        kth_p, kth_d, kth_i = self.gains[2]
        kp_limit, kd_limit, ki_limit = self.gains[3]
        self.ctrlLeeP.lambda_svm = 1000
        self.ctrlLeeP.lambdaa = self.gains[4]

        self.ctrlLeeP.Kpos_P.x = kpos_p
        self.ctrlLeeP.Kpos_P.y = kpos_p
        self.ctrlLeeP.Kpos_P.z = kpos_p

        self.ctrlLeeP.Kpos_D.x = kpos_d
        self.ctrlLeeP.Kpos_D.y = kpos_d
        self.ctrlLeeP.Kpos_D.z = kpos_d

        self.ctrlLeeP.Kpos_I.x = kpos_i
        self.ctrlLeeP.Kpos_I.y = kpos_i
        self.ctrlLeeP.Kpos_I.z = kpos_i

        self.ctrlLeeP.Kpos_P_limit = kp_limit
        self.ctrlLeeP.Kpos_I_limit = kd_limit
        self.ctrlLeeP.Kpos_D_limit = ki_limit


        self.ctrlLeeP.KR.x = kth_p
        self.ctrlLeeP.KR.y = kth_p
        self.ctrlLeeP.KR.z = kth_p
        self.ctrlLeeP.Komega.x = kth_d
        self.ctrlLeeP.Komega.y = kth_d
        self.ctrlLeeP.Komega.z = kth_d
        self.ctrlLeeP.KI.x = kth_i
        self.ctrlLeeP.KI.y = kth_i
        self.ctrlLeeP.KI.z = kth_i

        self.ctrlLeeP.K_q.x = kc_p
        self.ctrlLeeP.K_q.y = kc_p
        self.ctrlLeeP.K_q.z = kc_p
        self.ctrlLeeP.K_w.x = kc_d
        self.ctrlLeeP.K_w.y = kc_d
        self.ctrlLeeP.K_w.z = kc_d
        self.ctrlLeeP.KqIx = kc_i
        self.ctrlLeeP.KqIy = kc_i
        self.ctrlLeeP.KqIz = kc_i

    def setSetpoint(self):
        """Set the desired setpoint modes."""
        self.setpoint = cff.setpoint_t()
        self.setpoint.mode.x = cff.modeAbs
        self.setpoint.mode.y = cff.modeAbs
        self.setpoint.mode.z = cff.modeAbs
        self.setpoint.mode.quat = cff.modeAbs
        self.setpoint.mode.roll = cff.modeDisable
        self.setpoint.mode.pitch = cff.modeDisable
        self.setpoint.mode.yaw = cff.modeDisable
    
    def updateNeighbors(self, qpos, qvel, tendons=False):
        """Update the neighbors' states in the firmware controller."""
        for k, i in enumerate(self.team_ids):
            if not tendons:
                uav_pos, _, _, _ = self.getUAVState(qpos, qvel, i)
            else:
                uav_pos =  qpos[7 + 7*i: 7 + 7*i + 3]
            cff.state_set_position(self.state, k, k, uav_pos[0], uav_pos[1], uav_pos[2])
            cff.controller_lee_payload_set_attachement(
                self.ctrlLeeP, k, k, 0, 0, 0)

    def updateState(self, qpos, qvel, id, tendons=False):
        if not tendons:    
            """Update the drone and the payload from mujoco states without tendons in the firmware controller."""
            self.state.payload_pos.x = qpos[0]  # m
            self.state.payload_pos.y = qpos[1]  # m
            self.state.payload_pos.z = qpos[2]  # m
            self.state.payload_vel.x = qvel[0]  # m/s
            self.state.payload_vel.y = qvel[1]  # m/s
            self.state.payload_vel.z = qvel[2]  # m/s
            uav_pos, uav_vel, quat_uav, w = self.getUAVState(qpos, qvel, id)
            self.state.attitudeQuaternion.w = quat_uav[0]
            self.state.attitudeQuaternion.x = quat_uav[1]
            self.state.attitudeQuaternion.y = quat_uav[2]  
            self.state.attitudeQuaternion.z = quat_uav[3]
            rpy_state = rn.to_euler(quat_uav, convention="xyz")

            self.state.attitude.roll = np.degrees(rpy_state[0])
            self.state.attitude.pitch = np.degrees(-rpy_state[1])
            self.state.attitude.yaw = np.degrees(rpy_state[2])
            self.state.payload_quat.x = np.nan
            self.state.payload_quat.y = np.nan
            self.state.payload_quat.z = np.nan
            self.state.payload_quat.w = np.nan
            self.sensors.gyro.x = np.degrees(w[0])  # deg/s
            self.sensors.gyro.y = np.degrees(w[1])  # deg/s
            self.sensors.gyro.z = np.degrees(w[2])  # deg/s
        
            self.state.position.x = uav_pos[0]  # m
            self.state.position.y = uav_pos[1]  # m
            self.state.position.z = uav_pos[2]  # m
            self.state.velocity.x = uav_vel[0]  # m/s
            self.state.velocity.y = uav_vel[1]  # m/s
            self.state.velocity.z = uav_vel[2]  # m/s
        else:
            self.state.payload_pos.x = qpos[0]  # m
            self.state.payload_pos.y = qpos[1]  # m
            self.state.payload_pos.z = qpos[2]  # m
            self.state.payload_vel.x = qvel[0]  # m/s
            self.state.payload_vel.y = qvel[1]  # m/s
            self.state.payload_vel.z = qvel[2]  # m/s
            self.state.payload_quat.x = np.nan
            self.state.payload_quat.y = np.nan
            self.state.payload_quat.z = np.nan
            self.state.payload_quat.w = np.nan


            uav_qpos = qpos[7 + 7*id: 7 + 7*id + 7]
            uav_quat = qpos[7+7*id + 3 : 7+7*id + 7]
            self.state.position.x = uav_qpos[0]  # m
            self.state.position.y = uav_qpos[1]  # m
            self.state.position.z = uav_qpos[2]  # m
            self.state.attitudeQuaternion.w = uav_qpos[3]
            self.state.attitudeQuaternion.x = uav_qpos[4]
            self.state.attitudeQuaternion.y = uav_qpos[5]  
            self.state.attitudeQuaternion.z = uav_qpos[6]
            rpy_state = rn.to_euler(uav_quat, convention="xyz")

            self.state.attitude.roll = np.degrees(rpy_state[0])
            self.state.attitude.pitch = np.degrees(-rpy_state[1])
            self.state.attitude.yaw = np.degrees(rpy_state[2])

            uav_vel = qvel[6+6*id: 6 + 6*id + 3]
            # uav_vel = rn.rotate(uav_quat, uav_vel)
            self.state.velocity.x = uav_vel[0]  # m/s
            self.state.velocity.y = uav_vel[1]  # m/s
            self.state.velocity.z = uav_vel[2]  # m/s
            w_body = qvel[6 + 6*id + 3 : 6+6*id + 6]
            w = rn.rotate(uav_quat, w_body)
            self.sensors.gyro.x = np.degrees(w[0])  # deg/s
            self.sensors.gyro.y = np.degrees(w[1])  # deg/s
            self.sensors.gyro.z = np.degrees(w[2])  # deg/s



    def getUAVState(self, qpos, qvel, id):
        # this function is only used in case of rigid links
        quat_cable_in_world = qpos[7 + 8*id : 7 + 8*id + 4]
        quat_uav_in_cable_frame = qpos[7 + 8*id + 4 : 7 + 8*id + 8]

        quat_uav_in_world = rn.multiply(quat_cable_in_world, quat_uav_in_cable_frame)  # convert UAV quaternion to world frame
        
        w_uav_body = qvel[6 + 6*id + 3 : 6 + 6*id + 6]
        w = rn.rotate(quat_uav_in_world, w_uav_body)  # convert UAV angular velocity to world frame
        
        
        qc = rn.rotate(quat_cable_in_world, np.array([0, 0, -1]))
        wc_body = qvel[6+6*id : 6+6*id + 3]
        wc = rn.rotate(quat_cable_in_world, wc_body)  # convert cable angular velocity to world frame
        qc_dot = np.cross(wc, qc)
        
        uav_pos = np.array(qpos[0:3]) - self.l[id] * qc
        uav_vel = np.array(qvel[0:3]) - self.l[id] * qc_dot

        return uav_pos, uav_vel, quat_uav_in_world, w

    def comuteAngAcc(self, state, actions, ap, i, wpdot=None):
       # feedforward cableangacc (not used currently and only applicable for rigid links)
        action = actions[4 * i : 4 * i + 4]
        action*=self.u_nominal
        q = state[9 + 6 * self.num_robots + 7 * i : 9 + 6 * self.num_robots + 7 * i + 4]
        qc = state[9 + 6 * i : 9 + 6 * i + 3]
        q_rn = [q[3], q[0], q[1], q[2]]
        control = self.B0 @ action
        th = control[0]
        fu = np.array([0, 0, th])
        apgrav = ap + np.array([0, 0, 9.81])
        u_i = rn.rotate(q_rn, fu)
        qcqcT = qc.reshape(3, 1) @ qc.reshape(3, 1).T
        wcdot = 1 / self.l[i] * np.cross(qc, apgrav) - (
            1 / (self.ctrlLeeP.mass * self.l[i])
        ) * np.cross(qc, u_i)

        return wcdot

    def updateMuplanned(self, states_d, qpos, qvel, actions_d):
        # computes the desired forces of each cable
        grav = np.array([0, 0, 9.81])
        kpos_p, kpos_d, kpos_i = self.gains[0]
        state = np.array([qpos[0], qpos[1], qpos[2], qvel[0], qvel[1], qvel[2]])
        posp_e = states_d[0 : 3] - state[0 : 3]
        velp_e = states_d[3 : 6] - state[3 : 6]
        ap = states_d[6 : 9]
        accp = ap + grav
        F_ref = self.ctrlLeeP.mp * (accp + kpos_p * (posp_e) + kpos_d * (velp_e))
        qi_mat = np.zeros((3, self.num_robots))

        for k, i in enumerate(self.team_ids):
            qc = states_d[9 + 6 * i : 9 + 6 * i + 3]
            qi_mat[0:3, k] = -qc
        # Construct the problem
        n = self.num_robots
        T_vec = cp.Variable(n)
        objective = cp.Minimize(0.001 * cp.sum_squares(T_vec) + cp.sum_squares((qi_mat @ T_vec - F_ref)))
        constraints = [0.001*np.ones(n,)<= T_vec]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        T_vec = T_vec.value
        for k, i in enumerate(self.team_ids):
            qc = states_d[9 + 6 * i :  9 + 6 * i + 3]
            wc = states_d[9 + 6 * i + 3 :  9 + 6 * i + 6]
            qc_dot = np.cross(wc, qc)
            action = actions_d[4 * k : 4 * k + 4]
            action*=self.u_nominal
            control = self.B0 @ action
            self.ctrlLeeP.tau_ff.x = control[1]
            self.ctrlLeeP.tau_ff.y = control[2]
            self.ctrlLeeP.tau_ff.z = 0
            w_des = states_d[9 + 6 * self.num_robots + 4 : 9 + 6 * self.num_robots + 7]
            self.ctrlLeeP.omega_r.x = w_des[0]
            self.ctrlLeeP.omega_r.y = w_des[1]
            self.ctrlLeeP.omega_r.z = w_des[2]
            mu_planned = -T_vec[k] * qc
            
            wcdot = self.comuteAngAcc(states_d, actions_d, ap, i)
            cff.set_setpoint_qi_ref(self.setpoint, 
                k, k,
                mu_planned[0],
                mu_planned[1],
                mu_planned[2],
                # 0,0,0,
                qc_dot[0],
                qc_dot[1],
                qc_dot[2],
                0,0,0)
                # wcdot[0],
                # wcdot[1],
                # wcdot[2])

    def updateSetpoint(self, traj, qpos, qvel, actions=None):
        """Update the setpoint from the desired trajectory."""
        self.setpoint.position.x = traj[0]  # m
        self.setpoint.position.y = traj[1]  # m
        self.setpoint.position.z = traj[2]  # m 
        self.setpoint.velocity.x = traj[3]  # m/s
        self.setpoint.velocity.y = traj[4]  # m/s
        self.setpoint.velocity.z = traj[5]  # m/s
        self.setpoint.acceleration.x = traj[6]  # m/s^2
        self.setpoint.acceleration.y = traj[7]  # m/s^2
        self.setpoint.acceleration.z = traj[8]  # m/s^2

        if actions is not None:
            self.updateMuplanned(traj, qpos, qvel, actions)

    def getControl(self, id=0, tick=0):
        """Compute the motor forces command using the LeePayload controller."""
        cff.controllerLeePayload(self.ctrlLeeP, self.control, self.setpoint, self.sensors, self.state, tick)
        # payload_vel_prev is used for computation of numerical acceleration of the payload, in real flights this is estimated with a butterworth filter
        # this should move to a more convinient function but it doesn't affect the performance
        self.ctrlLeeP.payload_vel_prev.x = self.state.payload_vel.x 
        self.ctrlLeeP.payload_vel_prev.y = self.state.payload_vel.y 
        self.ctrlLeeP.payload_vel_prev.z = self.state.payload_vel.z 

        eta = np.array([self.control.thrustSI, self.control.torque[0], self.control.torque[1], self.control.torque[2]]) 
        mf = self.B0_inv @ eta
        u = np.clip(mf, 0.0, 0.15)     # keep within ctrlrange
        return u


class LeeController():
    def __init__(self, model_params):
        """Initialize the Lee controller."""
        self.ctrlLee = cff.controllerLee_t()
        cff.controllerLeeInit(self.ctrlLee)
        self.state = cff.state_t()
        self.sensors = cff.sensorData_t()
        self.control = cff.control_t()
        self.ctrlLee.mass = model_params["m"]
        self.setSetpoint()
        arm_length = 0.046  # m
        arm = 0.707106781 * arm_length
        t2t = 0.006  # thrust-to-torque ratio
        self.B0 = np.array([
            [1, 1, 1, 1],
            [-arm, -arm, arm, arm],
            [-arm, arm, arm, -arm],
            [-t2t, t2t, -t2t, t2t]
            ])
        self.B0_inv = np.linalg.inv(self.B0)
    
    def setSetpoint(self):
        """Set the desired setpoint modes."""
        self.setpoint = cff.setpoint_t()
        self.setpoint.mode.x = cff.modeAbs
        self.setpoint.mode.y = cff.modeAbs
        self.setpoint.mode.z = cff.modeAbs
        self.setpoint.mode.roll = cff.modeDisable
        self.setpoint.mode.pitch = cff.modeDisable
        self.setpoint.mode.yaw = cff.modeDisable

    def updateState(self, st):
        """Update the drone state in the firmware controller."""
        self.state.position.x = st[0]
        self.state.position.y = st[1]
        self.state.position.z = st[2]
        self.state.attitudeQuaternion.w = st[3]
        self.state.attitudeQuaternion.x = st[4]
        self.state.attitudeQuaternion.y = st[5]
        self.state.attitudeQuaternion.z = st[6]

        self.state.velocity.x = st[7]
        self.state.velocity.y = st[8]
        self.state.velocity.z = st[9]

        self.sensors.gyro.x = np.degrees(st[10])
        self.sensors.gyro.y = np.degrees(st[11])
        self.sensors.gyro.z = np.degrees(st[12])


    def updateSetpoint(self, traj):
        """Update the desired trajectory setpoint."""
        pos = traj[0:3]
        vel = traj[3:6]
        acc = traj[6:9]
        jerk = traj[9:12]
        snap = traj[12:15]

        self.setpoint.position.x = pos[0]
        self.setpoint.position.y = pos[1]
        self.setpoint.position.z = pos[2] 

        self.setpoint.velocity.x = vel[0]
        self.setpoint.velocity.y = vel[1]
        self.setpoint.velocity.z = vel[2]

        self.setpoint.acceleration.x = acc[0]
        self.setpoint.acceleration.y = acc[1]
        self.setpoint.acceleration.z = acc[2]

        self.setpoint.jerk.x = jerk[0]
        self.setpoint.jerk.y = jerk[1]
        self.setpoint.jerk.z = jerk[2]




    def getControl(self, id=0):
        """Compute the control command using the Lee controller."""
        cff.controllerLee(self.ctrlLee, self.control, self.setpoint, self.sensors, self.state, 0)
        eta = np.array([self.control.thrustSI, self.control.torque[0], self.control.torque[1], self.control.torque[2]])
        return self.B0_inv @ eta

class CFMujoco():
    def __init__(self, xml_path, name, sim_args):
        """Initialize the MuJoCo simulation."""
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.tendons = sim_args["tendons"]
        self.cam = mj.MjvCamera()
        self.cam.lookat[:] = [self.data.qpos[0], self.data.qpos[1], self.data.qpos[2] + .5]
        self.cam.azimuth = 0
        self.cam.elevation = -0
        self.cam.distance = 2.5
        self.payload = sim_args["payload"]
        self.opt = mj.MjvOption()
        self.actions_ff = True
        self.ghost_mode = False
        # Load trajectory and model parameters
        self.visualize    = sim_args["visualize"]
        self.model_params = sim_args["model_params"]
        self.traj_path    = sim_args["traj_path"]
        self.out_path     = sim_args["out_path"]
        # Determine number of robots
        self.num_robots = self.model_params.get("num_robots", 1)
        self.controller_list = []
        # record run index and initialize extra logs
        self.log_payload_pos = []
        self.log_payload_vel = []
        self.log_quad_pos    = []
        self.log_quad_vel    = []
        self.log_quad_rot    = []

        if self.visualize:
            if not glfw.init():
                raise RuntimeError("GLFW initialization failed!")
            self.window = glfw.create_window(2400, 1200, name, None, None)
            if not self.window:
                glfw.terminate()
                raise RuntimeError("GLFW window creation failed!")
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)

            mj.mjv_defaultCamera(self.cam)
            mj.mjv_defaultOption(self.opt)
            self.scene   = mj.MjvScene(self.model, maxgeom=10000)
            self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

            # --- setting up a ghost model to show dynoplan/dynobench trajectory -------------------------------
            self.ghost_model  = mj.MjModel.from_xml_path(xml_path)   # identical XML
            self.ghost_data   = mj.MjData(self.ghost_model)

            # 30 %-opaque green for every geom in the ghost
            self.ghost_model.geom_rgba[:] = (1.0, 0.0, 0.0, 0.5) 

            self.ghost_scene = mj.MjvScene(self.ghost_model, maxgeom=10000)
            self.ghost_opt   = mj.MjvOption() 
            self.ghost_opt.geomgroup[:] = 0     # hide everything
            self.ghost_opt.geomgroup[1] = 1     # show only group-1 geoms (the CFs)
            self.ghost_opt.geomgroup[2] = 1     # show only group-1 geoms (the CFs)
            self.ghost_opt.geomgroup[3] = 1     # show only group-1 geoms (the CFs)
            self.ghost_opt.geomgroup[5] = 1     # show only group-1 geoms (the CFs)

            self.ghost_context = mj.MjrContext(self.ghost_model, mj.mjtFontScale.mjFONTSCALE_150.value)                      
            # -------------------------------- default options ------------------------------------#

            # Install input callbacks
            glfw.set_key_callback(self.window, self.keyboard)
            glfw.set_cursor_pos_callback(self.window, self.mouse_move)
            glfw.set_mouse_button_callback(self.window, self.mouse_button)
            glfw.set_scroll_callback(self.window, self.scroll)

            self.button_left = False
            self.button_right = False
            self.last_x, self.last_y = 0, 0
            self.ref_markers_created = False
        else:
            # headless: no window/context
            self.window = None

        # log states and actions and time
        self.log_time   = []
        self.log_states = []
        self.log_actions = []
        self.plan_type = "dynoplan" #TODO: make this a parameter 
        if self.traj_path.endswith(".csv") and not self.payload: 
            # This is used for a single cf, reference trajectory (self.traj) contains:t px py pz vx vy vz ax ay az
            self.traj = loadcsv(self.traj_path) # Load trajectory from CSV file
            self.T = self.traj[-1, 0]
            self.ts = self.traj[:, 0]
            self.traj[:,0:-1] = self.traj[:,1::]
            self.traj[:, 2] += 0.5 # add 0.5 to the reference trajectory in the z-axis

        elif self.traj_path.endswith(".yaml") and self.payload: 
            # this is used for the payload, the reference trajectory is provided from dynoplan.
            # check function dynoplan_to_mujoco_states() to understand the state structure of dynoplan and the mapping to mujoco
            self.ghost_mode = True
            self.trajdata = loadyaml(self.traj_path)
            self.traj = np.array(self.trajdata["result"]["refstates"],dtype=np.float64)
            self.traj[:,2] += 0.5 # TODO: add 0.5 to the reference trajectory in the z-axis
            self.dynoplan_acttraj = np.array(self.trajdata["result"]["states"],dtype=np.float64) # this is the ghost states to compare dynobench model states with mujoco: check function dynoplan_to_mujoco_states()
            self.dynoplan_acttraj = np.insert(self.dynoplan_acttraj,6, np.zeros(len(self.dynoplan_acttraj)), axis=1) # add fake acc.x
            self.dynoplan_acttraj = np.insert(self.dynoplan_acttraj,7, np.zeros(len(self.dynoplan_acttraj)), axis=1) # add fake acc.y
            self.dynoplan_acttraj = np.insert(self.dynoplan_acttraj,8, np.zeros(len(self.dynoplan_acttraj)), axis=1) # add fake acc.z
            self.dynoplan_acttraj[:,2] += 0.5 # TODO: align the height shift with the real simulation 
            self.T = len(self.traj)
            if self.trajdata["result"]["actions_d"] and self.trajdata["result"]["actions_d"] is not None:
                self.planned_actions = np.array(self.trajdata["result"]["actions_d"],dtype=np.float64)
            else:
                self.planned_actions = None
                self.actions_ff = False
            self.ts = np.linspace(0, self.T, num=self.T-1)
        elif self.traj_path.endswith(".json") and self.payload:
            self.actions_ff = False
            self.plan_type = "payload_target_pos" #TODO: make this an arg 
            self.traj_data = load_start_goal_states(self.traj_path)
            self.model.opt.timestep = self.traj_data["dt"]
            
            self.T = 10 # sec
            self.start_state = self.traj_data["start_state"]
            self.track_traj = self.traj_data["trajectory"]
            self.ts = np.arange(0, self.T, step=self.model.opt.timestep)
            if self.track_traj is None:
                self.traj = np.zeros((self.ts.shape[0],12)) #pos vel acc snap
                payload_pos = self.traj_data["target_pos"]  
                self.traj[:, 0:3] = payload_pos

                print("Generating straight line trajectory for payload")
                # def generate_straight_line_trajectory(start, end, max_vel=1.5):
                #     """Generate a straight line trajectory between start and end points."""
                #     return np.linspace(start, end, num_points)
                print("Payload start position: ", self.traj_data["start_state"][0:3])
                print("Payload end position: ", self.traj_data["target_pos"])
                payload_start = self.traj_data["start_state"][0:3]
                payload_end = self.traj_data["target_pos"]
            

                
                generated_traj = generate_straight_line_trajectory(payload_start,
                                                                    payload_end, 
                                                                    max_vel=1.5, 
                                                                    max_acc=1.0,
                                                                    max_snap=3.0,
                                                                    dt=self.model.opt.timestep)
                print("Generated trajectory shape: ", generated_traj.shape)
                print("Generated trajectory: ", generated_traj[:5])

                print("last point of the generated trajectory: ", generated_traj[-1])

                # update the trajectory with the generated trajectory for the indices of the trajectory length

                self.traj[:generated_traj.shape[0], :] = generated_traj[:, 1:]  # skip the time column

                # save the generated trajectory to yaml file
                # with open("testtraj.yaml", 'w') as f:
                #     yaml.dump({"trajectory": self.traj.tolist()}, f)

            else:
                self.T = len(self.traj)
                self.traj = np.zeros((self.T,12)) #pos vel acc snap
                payload_st = self.traj_data["target_pos"]  
                self.traj[:, 0:3] = payload_st[0:3]
                self.traj[:, 3:6] = payload_st[3:6]
        else:
            print("Invalid trajectory file format")
        
        if self.payload:
            if self.plan_type == "payload_target_pos":
                self.model_params["m_payload"] = self.traj_data["m_payload"]            
                self.model_params["l_payload"] = [self.traj_data["l_payload"]]*self.num_robots
                self.model_params["plan_type"] = self.plan_type
            for i in range(self.num_robots):
                self.controller = LeePayloadController(self.model_params)
                self.controller_list.append(self.controller)
            self.controller.updateSetpoint(self.traj[0], self.data.qpos, self.data.qvel)
        
        else:
            self.controller = LeeController(self.model_params)
            self.controller.updateSetpoint(self.traj[0])

    def _set_ghost_state(self, ref_state):
        """Copy one saved dynoplan state into ghost_data (qpos, qvel)."""
        qpos_tmp, qvel_tmp = self.dynoplan_to_mujoco_states(dynoplan_st=ref_state, tendons=self.tendons)
        self.ghost_data.qpos[:] = qpos_tmp
        self.ghost_data.qvel[:] = qvel_tmp
        mj.mj_forward(self.ghost_model, self.ghost_data)        # quick copy from live->ghost

    def dynoplan_to_mujoco_states(self, dynoplan_st=None, tendons=False):
        """Conver the reference states from dynoplan to mujoco."""
        # reference states in dynoplan are in the form of [p0, v0, a0, qc1, wc1, qc2, wc2, ..., qcn, wcn], where n is the number of robots.
        # note that quat in dynoplan is in the form of [x, y, z, w], while in mujoco it is in the form of [w, x, y, z]
        # p0: payload position
        # v0: payload velocity
        # qci: cable unit vector pointing from the UAV to the payload
        # wci: cable angular velocity
        # quati: quaternion of the UAV in world frame
        # wi: angular velocity of the UAV in world frame
        # mujoco states if the model used has links are in the form of:
        #  self.data.qpos: [p0, quat0, quat_c1, quat1, quat_c2, quat2, ..., quat_cn, quatn]
        #  self.data.qvel: [v0, w0, wc1, wc2, ..., wcn]
        # quat0: quaternion of t he payload, in the point mass case it is always identity [1,0,0,0]
        # quat_ci: quaternion of the cable in the payload frame, in the point mass case it is always identity [1,0,0,0]
        # quati: quaternion of the UAV in the cable's frame 
    
        # mujoco states if the model used has TENDONS INSTEAD OF LINKS are in the form of:
        #  self.data.qpos: [p0, quat0,  quat1, ..., quatn]
        #  self.data.qvel: [v0, w0, w1, w2, ..., wn], note that w1 to wn are in the body frame
        # quat0: quaternion of t he payload, in the point mass case it is always identity [1,0,0,0]
        # quati: quaternion of the UAV in the cable's frame 

        data_tmp_qpos = self.data.qpos.copy()
        data_tmp_qvel = self.data.qvel.copy()
        if not tendons:
            data_tmp_qpos[0:3] = dynoplan_st[0:3]  # payload position
            data_tmp_qpos[3:7] = [1, 0, 0, 0] # payload quaternion (identity in point mass model)
            data_tmp_qvel[0:3] = dynoplan_st[3:6]  # payload velocity
            data_tmp_qvel[3:6] = np.zeros(3)  # payload angular velocity (not used in point mass model)
            data_tmp_qvel = np.zeros(6 + 6 * self.num_robots)  # initialize qvel with zeros
        
            for i in range(self.num_robots):
                dynoplan_cable_st = dynoplan_st[9 :9+6*self.num_robots]  # qci,wci
                dynoplan_uav_st = dynoplan_st[9+6*self.num_robots : 9+6*self.num_robots+7*self.num_robots]  # quati,wi   
                quat_cable = (rn.vector_vector_rotation(np.array([0,0,-1]), dynoplan_cable_st[6*i : 6*i+3]))
                quat_uav = reorder_quat(dynoplan_uav_st[7*i : 7*i+4])
                quat_uav_cable =  rn.multiply(rn.conjugate(quat_cable),quat_uav)  # convert UAV quaternion to cable frame
                data_tmp_qpos[7+8*i : 7+8*i+4] = quat_cable.copy()
                data_tmp_qpos[7+8*i+4: 7+8*i+8] = quat_uav_cable.copy() #TODO: adjust UAV quaternion to cable frame

                w_cable = dynoplan_cable_st[6*i+3 : 6*i+6]
                w_uav = dynoplan_uav_st[7*i+4 : 7*i+7]

                data_tmp_qvel[6+6*i : 6+6*i+3]   =  rn.rotate(quat_cable, w_cable)
                data_tmp_qvel[6+6*i+3 : 6+6*i+6] = rn.rotate(rn.multiply(quat_cable, quat_uav_cable), w_uav)
                
        elif tendons:
                data_tmp_qpos[0:3] = dynoplan_st[0:3] # payload position, TODO: adjust z position for payload automatically
                data_tmp_qpos[3:7] = [1, 0, 0, 0] # payload quaternion (identity in point mass model)
                data_tmp_qvel[0:3] = dynoplan_st[3:6]  # payload velocity
                data_tmp_qvel[3:6] = np.zeros(3)  # payload angular velocity (not used in point mass model)
                data_tmp_qvel = np.zeros(6 * (self.num_robots+1))  # initialize qvel with zeros
                l_c = 0.5 # TODO: update length of cables
                for i in range(self.num_robots):
                    dynoplan_cable_st = dynoplan_st[9 :9+6*self.num_robots]  # qci,wci
                    dynoplan_uav_st = dynoplan_st[9+6*self.num_robots : 9+6*self.num_robots+7*self.num_robots]  # quati,wi   
                    qi = dynoplan_cable_st[6*i : 6*i+3] # cable direction
                    quat_cable = (rn.vector_vector_rotation(np.array([0,0,-1]), qi))
                    quat_uav = reorder_quat(dynoplan_uav_st[7*i : 7*i+4])
                    uav_i_pos = data_tmp_qpos[0:3] - l_c*qi
                    data_tmp_qpos[7 + 7*i : 7 + 7*i + 3] = uav_i_pos
                    data_tmp_qpos[7 + 7*i + 3 : 7 + 7*i + 7] = quat_uav
        else:
            print("Model is wrong!")
            exit()
        return data_tmp_qpos, data_tmp_qvel


    def setInitState(self):
        """Set the initial state of the drone."""
        if self.payload:
            if self.plan_type == "dynoplan":
                qpos_tmp, qvel_tmp = self.dynoplan_to_mujoco_states(dynoplan_st=self.traj[0],tendons=self.tendons)
                self.data.qpos[:] = qpos_tmp 
                self.data.qvel[:] = qvel_tmp
            elif self.plan_type == "polynomial":
                self.data.qpos[:] = [self.traj[0,0], self.traj[0,1], self.traj[0,2], 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
                self.data.vel[:] = np.zeros(6 + 6 * self.num_robots)  # initialize qvel with zeros
            elif self.plan_type == "payload_target_pos":
                self.data.qpos[:] = self.start_state[0:self.model.nq]
                self.data.qvel[:] = self.start_state[self.model.nq:self.model.nq+self.model.nv]
        else:
            print("Setting initial state for UAV without payload")
            self.data.qpos[:] = [self.traj[0,0], self.traj[0,1], self.traj[0,2], 1, 0, 0, 0]
            self.data.qvel[:] = [0, 0, 0, 0, 0, 0]
            self.controller.updateState(np.concatenate((self.data.qpos, self.data.qvel)))
        self.log_states.append(np.concatenate((self.data.qpos, self.data.qvel)).tolist())

    def keyboard(self, window, key, scancode, action, mods):
        """Handle keyboard events."""
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

    def mouse_button(self, window, button, action, mods):
        """Handle mouse button events."""
        if action == glfw.PRESS:
            self.button_left = (button == glfw.MOUSE_BUTTON_LEFT)
            self.button_right = (button == glfw.MOUSE_BUTTON_RIGHT)
            self.last_x, self.last_y = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            self.button_left = self.button_right = False

    def mouse_move(self, window, xpos, ypos):
        """Handle mouse movement for camera control."""
        dx = xpos - self.last_x
        dy = ypos - self.last_y
        self.last_x, self.last_y = xpos, ypos

        if self.button_left:
            mj.mjv_moveCamera(self.model, mj.mjtMouse.mjMOUSE_ROTATE_H, dx / 100, dy / 100, self.scene, self.cam)
        elif self.button_right:
            mj.mjv_moveCamera(self.model, mj.mjtMouse.mjMOUSE_ZOOM, dx / 100, dy / 100, self.scene, self.cam)

    def scroll(self, window, xoffset, yoffset):
        """Handle scroll events for zooming."""
        mj.mjv_moveCamera(self.model, mj.mjtMouse.mjMOUSE_ZOOM, 0, yoffset / 10, self.scene, self.cam)

    def simulate(self):
        """Run the simulation loop, headless or interactive."""
        # ---- HEADLESS MODE ----
        if not self.visualize:
            self.setInitState()
            print(f"Headless simulation for {len(self.ts)*self.model.opt.timestep:.2f} sec, logging to JSON.")
            for k, t in enumerate(self.ts):
                self.log_time.append(t)
                if self.payload:
                    for id in range(self.num_robots):
                        self.controller_list[id].team_ids.remove(id)
                        self.controller_list[id].team_ids.insert(0, id) 
                        if self.actions_ff:
                            self.controller_list[id].updateSetpoint(self.traj[k], self.data.qpos, self.data.qvel, actions=self.planned_actions[k])
                        else: 
                            self.controller_list[id].updateSetpoint(self.traj[k], self.data.qpos, self.data.qvel, actions=None)
                        self.controller_list[id].updateState(self.data.qpos, self.data.qvel, id, tendons=self.tendons)
                        self.controller_list[id].updateNeighbors(self.data.qpos, self.data.qvel, tendons=self.tendons)
                        force = self.controller_list[id].getControl(id=id, tick=k)
                        self.data.ctrl[4*id:4*id+4] = force
                else:
                    force = self.controller.getControl(id=0)
                    self.data.ctrl[0:4] = force

                mj.mj_step(self.model, self.data)
                # Log payload state
                self.log_payload_pos.append(self.data.qpos[0:3].tolist())
                self.log_payload_vel.append(self.data.qvel[0:3].tolist())
                # Quad logs (handle both rigid‐link and tendon cases)
                qp, qv, qr = [], [], []
                if self.tendons:
                    # when using tendons, controller.state holds the true world-frame UAV state
                    for ctrl in self.controller_list:
                        # position
                        p = [
                            ctrl.state.position.x,
                            ctrl.state.position.y,
                            ctrl.state.position.z,
                        ]
                        # linear velocity
                        v = [
                            ctrl.state.velocity.x,
                            ctrl.state.velocity.y,
                            ctrl.state.velocity.z,
                        ]
                        # quaternion (w, x, y, z)
                        quat = [
                            ctrl.state.attitudeQuaternion.w,
                            ctrl.state.attitudeQuaternion.x,
                            ctrl.state.attitudeQuaternion.y,
                            ctrl.state.attitudeQuaternion.z,
                        ]
                        qp.append(p)
                        qv.append(v)
                        qr.append(quat)
                else:
                    # rigid‐link case: recompute via getUAVState
                    for i in range(self.num_robots):
                        p, v, quat, _ = self.controller_list[i].getUAVState(
                            self.data.qpos, self.data.qvel, i
                        )
                        qp.append(p.tolist())
                        qv.append(v.tolist())
                        qr.append(list(quat))

                self.log_quad_pos.append(qp)
                self.log_quad_vel.append(qv)
                self.log_quad_rot.append(qr)
                # state/action logs
                self.log_states.append(np.concatenate((self.data.qpos, self.data.qvel)).tolist())
                self.log_actions.append(self.data.ctrl.tolist())

            if self.out_path:
                log_traj_data = load_json(self.traj_path)
                log_traj_data["states"]  = self.log_states
                log_traj_data["actions"] = self.log_actions
                log_traj_data["time"]    = self.log_time
                log_traj_data["payload_pos"] = self.log_payload_pos
                log_traj_data["payload_linvel"] = self.log_payload_vel
                log_traj_data["quad_pos"] = self.log_quad_pos
                log_traj_data["quad_linvel"] = self.log_quad_vel
                log_traj_data["quad_rot"] = self.log_quad_rot

                save_json(self.out_path, log_traj_data)
            return

        # ---- INTERACTIVE MODE (unchanged rendering loop) ----
        while not glfw.window_should_close(self.window):
            self.cam.lookat[:] = [self.data.qpos[0]-0.5, self.data.qpos[1], self.data.qpos[2] + 0.5]
            self.cam.azimuth = 0
            self.cam.elevation = -0
            self.cam.distance = 2.5
            self.setInitState() # set initial state before simulating
            print("Simulation started with trajectory length:", len(self.traj)*self.model.opt.timestep, "sec")
            for k, t in enumerate(self.ts):
                self.log_time.extend([t])
                if self.payload:
                    for id in range(self.num_robots):
                        self.controller_list[id].team_ids.remove(id)
                        self.controller_list[id].team_ids.insert(0, id) 
                        if self.actions_ff:
                            self.controller_list[id].updateSetpoint(self.traj[k], self.data.qpos, self.data.qvel, actions=self.planned_actions[k])
                        else: 
                            self.controller_list[id].updateSetpoint(self.traj[k], self.data.qpos, self.data.qvel, actions=None)
                        self.controller_list[id].updateState(self.data.qpos, self.data.qvel, id, tendons=self.tendons)
                        self.controller_list[id].updateNeighbors(self.data.qpos, self.data.qvel, tendons=self.tendons)
                        force = self.controller_list[id].getControl(id=id, tick=k) # force of each motor
                        self.data.ctrl[4*id:4*id+4] = np.array(force)
                else:
                    self.controller.updateSetpoint(self.traj[k])
                    self.controller.updateState(np.concatenate((self.data.qpos, self.data.qvel)))
                    force = self.controller.getControl(id=0)
                    self.data.ctrl[0:4] = np.array(force)
                mj.mj_step(self.model, self.data)
                # get framebuffer viewport
                viewport_width, viewport_height = glfw.get_framebuffer_size(
                    self.window)
                viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

                # Update scene and render
                mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                    mj.mjtCatBit.mjCAT_ALL.value, self.scene)

                if self.payload and self.ghost_mode:
                    self._set_ghost_state(self.dynoplan_acttraj[k])
                    mj.mjv_updateScene(self.ghost_model, self.ghost_data, self.ghost_opt,
                                    None, self.cam, mj.mjtCatBit.mjCAT_ALL.value,
                                    self.ghost_scene)
                    mj.mjv_addGeoms(self.ghost_model, self.ghost_data, self.ghost_opt,
                                    mj.MjvPerturb(), mj.mjtCatBit.mjCAT_ALL.value, self.scene)

                # # 3) draw both: ghost first (blue), live second (solid)
                mj.mjr_render(viewport, self.scene,       self.context)        # live second

                # swap OpenGL buffers (blocking call due to v-sync)
                glfw.swap_buffers(self.window)

                # process pending GUI events, call GLFW callbacks
                glfw.poll_events()
                self.log_states.append(np.concatenate((self.data.qpos, self.data.qvel)).tolist())
                self.log_actions.append(self.data.ctrl.tolist())
            if not self.visualize and self.out_path is not None: 
                log_traj_data = load_json(self.traj_path)
                log_traj_data["states"] = self.log_states
                log_traj_data["actions"] = self.log_actions
                log_traj_data["time"] = self.log_time
                save_json(self.out_path, log_traj_data)           
            if not self.visualize:
                glfw.set_window_should_close(self.window, True)
        glfw.terminate()
    
      
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_path', required=True, type=str, help="Path to trajectory CSV file or YAML file")
    parser.add_argument('--out_path', required=False, default=None, type=str, help="Path to trajectory CSV file or YAML file")
    parser.add_argument('--models_path', required=True, type=str, help="Path to model parameters YAML file")
    parser.add_argument('--mj', required=True, type=str, help="Path to MuJoCo xml file")
    parser.add_argument("-p", "--payload", action="store_true")
    parser.add_argument("-t", "--tendons", action="store_true", help="this is specific to the payload case. If it's true then the simulation assume you have tendons between the payload and the quadrotor, otherwise they are rigid links")
    parser.add_argument("-v", "--visualize", action="store_true", help="enable/disable visualizer, by default it is disabled")
    args = parser.parse_args()
    
    # Load model parameters
    model_params = loadyaml(args.models_path)
    sim_args = {"model_params": model_params, "traj_path": args.traj_path, "payload": args.payload, "tendons": args.tendons, "visualize": args.visualize, "out_path": args.out_path}
    # Initialize and run simulation
    xml_path = args.mj
    sim = CFMujoco(xml_path, "Crazyflie Simulation", sim_args)
    sim.simulate()

if __name__ == "__main__":
    main()
