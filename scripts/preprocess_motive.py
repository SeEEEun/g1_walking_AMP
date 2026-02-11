#!/usr/bin/env python3
import math, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os

# 파일 경로 설정 (본인의 환경에 맞게 수정)
CSV_PATH = "../data/g1_walking.csv"
URDF_PATH = "../assets/g1_12dof.urdf"

# Motive(Y-up) -> Isaac(Z-up) 변환 (Motive: x,y,z -> Isaac: x, z, -y)
APPLY_YUP_TO_ZUP = True

# =========================
# Quaternion / Rotation utils
# =========================
def quat_normalize_xyzw(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0, 0, 0, 1.0], dtype=float)
    return q / n

def quat_conj_xyzw(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=float)

def quat_inv_xyzw(q):
    q = quat_normalize_xyzw(q)
    return quat_conj_xyzw(q)

def quat_mul_xyzw(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.array([x, y, z, w], dtype=float)

def quat_to_R_xyzw(q):
    x, y, z, w = quat_normalize_xyzw(q)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz), 2*(xz+wy)],
        [2*(xy+wz), 1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy)]
    ], dtype=float)

def wrap_pi(a):
    return (a + math.pi) % (2.0*math.pi) - math.pi

def make_quat_continuous(Q_xyzw):
    """프레임마다 q와 -q가 섞이면 점프 생김 -> dot < 0 이면 부호 뒤집어 연속화"""
    Q = Q_xyzw.astype(float).copy()
    n = np.linalg.norm(Q, axis=1, keepdims=True)
    Q = Q / np.maximum(n, 1e-12)
    for i in range(1, len(Q)):
        if float(np.dot(Q[i], Q[i-1])) < 0.0:
            Q[i] *= -1.0
    return Q

def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ], dtype=float)

def R_to_quat_xyzw(R):
    m = R
    tr = float(np.trace(m))
    if tr > 0:
        S = math.sqrt(tr+1.0)*2
        w = 0.25*S
        x = (m[2,1]-m[1,2])/S
        y = (m[0,2]-m[2,0])/S
        z = (m[1,0]-m[0,1])/S
    else:
        if (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            S = math.sqrt(1.0+m[0,0]-m[1,1]-m[2,2])*2
            w = (m[2,1]-m[1,2])/S
            x = 0.25*S
            y = (m[0,1]+m[1,0])/S
            z = (m[0,2]+m[2,0])/S
        elif m[1,1] > m[2,2]:
            S = math.sqrt(1.0+m[1,1]-m[0,0]-m[2,2])*2
            w = (m[0,2]-m[2,0])/S
            x = (m[0,1]+m[1,0])/S
            y = 0.25*S
            z = (m[1,2]+m[2,1])/S
        else:
            S = math.sqrt(1.0+m[2,2]-m[0,0]-m[1,1])*2
            w = (m[1,0]-m[0,1])/S
            x = (m[0,2]+m[2,0])/S
            y = (m[1,2]+m[2,1])/S
            z = 0.25*S
    return quat_normalize_xyzw([x, y, z, w])

def angle_about_axis_from_R(R, axis):
    """R에서 axis 방향 회전 성분만 뽑아서 angle(rad) 반환"""
    q = R_to_quat_xyzw(R)
    v = np.array(q[:3], dtype=float)
    w = float(q[3])
    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    s = float(np.dot(v, a))
    ang = 2.0 * math.atan2(s, w)
    return wrap_pi(ang)

def finite_diff(q, dt):
    """중앙차분으로 속도 계산"""
    dq = np.zeros_like(q)
    dq[1:-1] = (q[2:] - q[:-2])/(2*dt)
    dq[0] = (q[1] - q[0])/dt
    dq[-1] = (q[-1] - q[-2])/dt
    return dq

def unwrap_angles(q):
    return np.unwrap(q, axis=0)

# =========================
# Motive CSV parser
# =========================
def read_motive_csv(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header_rows = [next(reader) for _ in range(8)]

    name_row = header_rows[3]
    prop_row = header_rows[6]
    comp_row = header_rows[7]

    cols = []
    for i in range(len(comp_row)):
        if i == 0:
            cols.append("Frame")
        elif i == 1:
            cols.append("Time")
        else:
            rb = name_row[i] if name_row[i] else "unknown"
            prop = prop_row[i] if prop_row[i] else "Value"
            comp = comp_row[i] if comp_row[i] else f"c{i}"
            cols.append(f"{rb}|{prop}|{comp}")
    df = pd.read_csv(path, skiprows=8, header=None, names=cols)
    return df

def get_quat(df, rb):
    cols = [f"{rb}|Rotation|X", f"{rb}|Rotation|Y", f"{rb}|Rotation|Z", f"{rb}|Rotation|W"]
    return df[cols].to_numpy(float)

def get_pos(df, rb):
    cols = [f"{rb}|Position|X", f"{rb}|Position|Y", f"{rb}|Position|Z"]
    return df[cols].to_numpy(float)

# =========================
# URDF Axis Parser
# =========================
def joint_axis_parent(urdf_path, joint_name):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    j = None
    for x in root.findall("joint"):
        if x.attrib.get("name") == joint_name:
            j = x
            break
    if j is None:
        raise KeyError(f"URDF에서 joint '{joint_name}' 못 찾음")

    axis_el = j.find("axis")
    axis = np.array([1.0, 0.0, 0.0])
    if axis_el is not None and axis_el.attrib.get("xyz"):
        axis = np.array([float(v) for v in axis_el.attrib["xyz"].split()], dtype=float)

    origin_el = j.find("origin")
    rpy = (0.0, 0.0, 0.0)
    if origin_el is not None and origin_el.attrib.get("rpy"):
        rpy = tuple(float(v) for v in origin_el.attrib["rpy"].split())

    R = rpy_to_R(*rpy)
    a = R @ axis
    a = a / (np.linalg.norm(a) + 1e-12)
    return a

# =========================
# Coordinate conversion
# =========================
def yup_to_zup_pos(p_xyz):
    x, y, z = p_xyz.T
    return np.vstack([x, z, -y]).T

def yup_to_zup_quat_xyzw(q_xyzw):
    ang = -math.pi/2
    q_conv = np.array([math.sin(ang/2), 0.0, 0.0, math.cos(ang/2)], dtype=float)
    Q = q_xyzw.copy()
    out = np.zeros_like(Q)
    for i in range(len(Q)):
        out[i] = quat_mul_xyzw(q_conv, Q[i])
    return out

# =========================
# Main Processing
# =========================
def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return
    if not os.path.exists(URDF_PATH):
        print(f"Error: URDF file not found at {URDF_PATH}")
        return

    df = read_motive_csv(CSV_PATH)
    t = df["Time"].astype(float).to_numpy()
    dt = float(np.median(np.diff(t)))
    fps = int(round(1.0/dt))

    # --- Root Pose ---
    root_pos = get_pos(df, "g1_torso")
    root_quat = get_quat(df, "g1_torso")

    # --- Leg Link Quats ---
    q_torso = make_quat_continuous(root_quat)
    q_Lth = make_quat_continuous(get_quat(df, "g1_left_thigh"))
    q_Lsh = make_quat_continuous(get_quat(df, "g1_left_shin"))
    q_Lft = make_quat_continuous(get_quat(df, "g1_left_foot"))
    q_Rth = make_quat_continuous(get_quat(df, "g1_right_thigh"))
    q_Rsh = make_quat_continuous(get_quat(df, "g1_right_shin"))
    q_Rft = make_quat_continuous(get_quat(df, "g1_right_foot"))

    # --- Coordinate Conversion ---
    if APPLY_YUP_TO_ZUP:
        root_pos = yup_to_zup_pos(root_pos)
        q_torso = yup_to_zup_quat_xyzw(q_torso)
        q_Lth = yup_to_zup_quat_xyzw(q_Lth)
        q_Lsh = yup_to_zup_quat_xyzw(q_Lsh)
        q_Lft = yup_to_zup_quat_xyzw(q_Lft)
        q_Rth = yup_to_zup_quat_xyzw(q_Rth)
        q_Rsh = yup_to_zup_quat_xyzw(q_Rsh)
        q_Rft = yup_to_zup_quat_xyzw(q_Rft)

    # --- Joint Axes ---
    joints = [
        "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
    ]
    axes = {name: joint_axis_parent(URDF_PATH, name) for name in joints}

    # --- Helper Logic ---
    def axis_angle_R(axis_world, ang):
        axis_world = axis_world/(np.linalg.norm(axis_world)+1e-12)
        x, y, z = axis_world
        c = math.cos(ang); s = math.sin(ang); C = 1-c
        return np.array([
            [c+x*x*C, x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, c+y*y*C, y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, c+z*z*C]
        ], dtype=float)

    def solve_chain(R_target, axes_list):
        R_rem = R_target.copy()
        R_acc = np.eye(3)
        angles = []
        for a_parent in axes_list:
            a0 = (R_acc @ a_parent)
            ang = angle_about_axis_from_R(R_rem, a0)
            angles.append(ang)
            Rj = axis_angle_R(a0, ang)
            R_rem = Rj.T @ R_rem
            R_acc = Rj @ R_acc
        return angles

    # --- Compute 12DOF ---
    N = len(t)
    q_dof = np.zeros((N, 12), dtype=float)

    for i in range(N):
        R_torso = quat_to_R_xyzw(q_torso[i])
        
        # Left Leg Hip
        R_Lth = quat_to_R_xyzw(q_Lth[i])
        L_yaw, L_roll, L_pitch = solve_chain(R_torso.T @ R_Lth, [axes["left_hip_yaw_joint"], axes["left_hip_roll_joint"], axes["left_hip_pitch_joint"]])
        
        # Right Leg Hip
        R_Rth = quat_to_R_xyzw(q_Rth[i])
        R_yaw, R_roll, R_pitch = solve_chain(R_torso.T @ R_Rth, [axes["right_hip_yaw_joint"], axes["right_hip_roll_joint"], axes["right_hip_pitch_joint"]])

        # Knee
        L_knee = angle_about_axis_from_R(R_Lth.T @ quat_to_R_xyzw(q_Lsh[i]), axes["left_knee_joint"])
        R_knee = angle_about_axis_from_R(R_Rth.T @ quat_to_R_xyzw(q_Rsh[i]), axes["right_knee_joint"])

        # Ankle
        R_Lft = quat_to_R_xyzw(q_Lft[i])
        R_Rft = quat_to_R_xyzw(q_Rft[i])
        L_ap, L_ar = solve_chain(quat_to_R_xyzw(q_Lsh[i]).T @ R_Lft, [axes["left_ankle_pitch_joint"], axes["left_ankle_roll_joint"]])
        R_ap, R_ar = solve_chain(quat_to_R_xyzw(q_Rsh[i]).T @ R_Rft, [axes["right_ankle_pitch_joint"], axes["right_ankle_roll_joint"]])

        q_dof[i,:] = [L_yaw, L_roll, L_pitch, L_knee, L_ap, L_ar, R_yaw, R_roll, R_pitch, R_knee, R_ap, R_ar]

    q_dof = unwrap_angles(q_dof)
    dq_dof = finite_diff(q_dof, dt)

    # --- Plotting ---
    plt.figure()
    plt.plot(t, q_dof[:, 3], label="left_knee_pitch(rad)")
    plt.plot(t, q_dof[:, 9], label="right_knee_pitch(rad)")
    plt.xlabel("Time (s)"); plt.ylabel("rad")
    plt.legend(); plt.tight_layout()
    plt.savefig("knee_pitch_rad.png", dpi=160)

    # --- Saving ---
    cols = ["L_hip_y","L_hip_r","L_hip_p","L_knee","L_ank_p","L_ank_r","R_hip_y","R_hip_r","R_hip_p","R_knee","R_ank_p","R_ank_r"]
    out = pd.DataFrame(q_dof, columns=cols)
    out.insert(0, "time_s", t)
    out.to_csv("g1_12dof_rad.csv", index=False)

    traj = {
        "fps": fps, "dt": dt, "time_s": t,
        "root_pos": root_pos.astype(np.float32),
        "root_rot": make_quat_continuous(q_torso).astype(np.float32),
        "dof_pos": q_dof.astype(np.float32),
        "dof_vel": dq_dof.astype(np.float32),
        "dof_names": np.array(cols, dtype=object),
    }
    np.save("../assets/amp/g1_amp_traj.npy", traj, allow_pickle=True)
    print(f"[OK] Processed {len(t)} frames. saved: g1_amp_traj.npy")

if __name__ == "__main__":
    main()
