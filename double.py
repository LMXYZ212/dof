#!/usr/bin/env python3
from calibration.calib_models import CalibrationModel6RComplNl
from calibration.calib_solvers import CalibrationSolverIpopt
from utils.utils import *
import numpy as np

import torch, torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import os

# ---------- DH / Kin  ----------
NDOF = 6
KINVEC = [
    [0, 0, 0.167899996],
    [0, -0.060899999, 0.0970999971],
    [0, 0, 0.444000006],
    [0.112999998, 0.060899999, 0.109999999],
    [0.356999993, 0.056499999, 0],
    [0.101000004, -0.056499999, 0.0799999982],
]

TCP = [0.135, -0.09, -0.07]




# joint limit
JOINT_LIMITS = (np.ones((NDOF, 2)) * np.array([[-180, 180]])).astype(float) / 180 * np.pi

def make_delta_q(q: np.ndarray, group_len: int = 120) -> np.ndarray:
    """
    计算 Δq，并在每组首帧处补 0
    ----------
    q.shape = (N, 6)
    返回 Δq.shape = (N, 6)
    """
    dq = np.diff(q, axis=0, prepend=q[:1])  # 差分并补第1行
    if group_len > 0:
        group_start_idx = np.arange(0, len(dq), group_len)
        dq[group_start_idx] = 0.0            # 每组首帧 Δq 清零
    return dq


def print_w_pi_grouped(w_pi_DM, pi_beta_all, pi_enabled):
    ptr = 0
    for j in range(6):
        if not pi_enabled[j]:
            print(f"Joint {j+1}: (disabled)")
            continue
        beta_list = pi_beta_all[j]
        n = len(beta_list)
        w_j = cs.vertsplit(w_pi_DM[ptr:ptr + n])  # ✅ 修复点
        print(f"Joint {j+1}:")
        for beta, w in zip(beta_list, w_j):
            print(f"    β={beta:.3f} → w={float(w):.5f}")
        ptr += n

def main() -> None:
   
        
    compliance_model = "Cubic"       # "Lin" / "Quad" / "Cubic"
    max_samples = 7200
    validation_ratio = 0.5           

    # ------------------------------------------------------------------

    # pkl file
    load_fct_file_high = (
        "load_func_holder_5kg.pkl"
    )
    load_fct_file_low = (
        "load_func_holder_only.pkl"
    )
    
    data_file_high = os.path.join(
        "data",
        "1500-multidir_JT120x60_5kg_v1000.tri"
    )
    
    data_file_low = os.path.join(
        "data",
        "1500-multidir_JT120x60_0kg_v1000.tri"
    )


    with open(load_fct_file_high, "rb") as f:
        wrench_fct_high = pickle.load(f)
        
    with open(load_fct_file_low, "rb") as f:
        wrench_fct_low = pickle.load(f)



        # ------ tri file ------
    q_all_high, tcp_all_high = parse_tri_file(data_file_high)       # q_all:(N,6)  tcp_all:(N,3)
    q_all_high, tcp_all_high = np.asarray(q_all_high), np.asarray(tcp_all_high)
    
    q_all_low, tcp_all_low = parse_tri_file(data_file_low)       # q_all:(N,6)  tcp_all:(N,3)
    q_all_low, tcp_all_low= np.asarray(q_all_low), np.asarray(tcp_all_low)


    
    q_all_high, tcp_all_high = q_all_high[:max_samples], tcp_all_high[:max_samples]
    q_all_low,  tcp_all_low  = q_all_low[:max_samples], tcp_all_low[:max_samples]

    split_high = int((1 - validation_ratio) * len(q_all_high))
    q_calib_high, q_valid_high = q_all_high[:split_high], q_all_high[split_high:]
    tcp_calib_high, tcp_valid_high = tcp_all_high[:split_high], tcp_all_high[split_high:]
    
    split_low = int((1 - validation_ratio) * len(q_all_low))
    q_calib_low, q_valid_low = q_all_low[:split_low], q_all_low[split_low:]
    tcp_calib_low, tcp_valid_low = tcp_all_low[:split_low], tcp_all_low[split_low:]
    
    def print_err(tag, mean_err, max_err):
        print(f"{tag:<10s}  mean={mean_err*1000:.4f} mm   max={max_err*1000:.4f} mm")

    def make_switch(geom_flags, keep_comp=False, compliance_model="Lin"):
        if compliance_model == "Lin":
            num_comp_param = 1
        elif compliance_model == "Quad":
            num_comp_param = 2
        elif compliance_model == "Cubic":
            num_comp_param = 3
        else:
            raise ValueError(f"Unknown compliance_model: {compliance_model}")

        return geom_flags + ([True] * num_comp_param if keep_comp else [False] * num_comp_param)

    print(f"[INFO] Calib poses high: {len(q_calib_high)}   Valid poses high: {len(q_valid_high)}")
    print(f"[INFO] Calib poses low: {len(q_calib_low)}   Valid poses low: {len(q_valid_low)}")

    compliance_set = compliance_model
    # ------------ Error para -----------------
    

    order_map = {"Lin": 1, "Quad": 2, "Cubic": 3}
    if compliance_model not in order_map:
        raise ValueError("Unsupported compliance model: " + compliance_model)
    



    # --------- 6R-3D---------
    compliance_model = 'Lin'
    
    num_comp_param = order_map[compliance_model]
    model_high = CalibrationModel6RComplNl(
        kinvec=KINVEC, tcp=TCP, load_fct=wrench_fct_high, comp_model=compliance_model
    )
    model_low = CalibrationModel6RComplNl(
        kinvec=KINVEC, tcp=TCP, load_fct=wrench_fct_low, comp_model=compliance_model
    )

    # ---------  para lock ----------
    
    num_params_per_ax = 6 + num_comp_param   # joint 6 kin and compliance
    

    # —— each joint ——

    sw_ax1 = make_switch([True, True, True, True, True, True], keep_comp=False, compliance_model=compliance_model)
    sw_ax2 = make_switch([True, False, False, True, True, False], keep_comp=True, compliance_model=compliance_model)
    sw_ax3 = make_switch([False, False, True, True, True, True], keep_comp=True,compliance_model=compliance_model)
    sw_ax4 = make_switch([False, True, True, True, False, True], keep_comp=True,compliance_model=compliance_model)
    sw_ax5 = make_switch([True, False, True, False, True, True], keep_comp=True,compliance_model=compliance_model)
    sw_ax6 = make_switch([False, True, True, True, False, False], keep_comp=False,compliance_model=compliance_model)
    sw_tool = make_switch([True, False, True, False, False, False], keep_comp=False,compliance_model=compliance_model)


    err_par_switch = [
        sw_ax1, sw_ax2, sw_ax3, sw_ax4, sw_ax5, sw_ax6, sw_tool
    ]

    

    model_high.set_error_par_switch(err_par_switch)
    model_low.set_error_par_switch(err_par_switch)

    # --------- initial reeor para ---------------
    init_guess = np.zeros((7 * num_params_per_ax,))

    solver_double = CalibrationSolverIpopt()
    solved_params_joint = solver_double.solve_double_calibration(
        model_high=model_high,
        model_low=model_low,
        q_high=q_calib_high,
        meas_high=tcp_calib_high,
        q_low=q_calib_low,
        meas_low=tcp_calib_low,
        comp_model=compliance_model,
        initial_guess=np.zeros(7 * num_params_per_ax),
    )    
    
    if compliance_set == 'Lin':
        mean_joint_high, max_joint_high = model_high.get_error(q_valid_high, tcp_valid_high, solved_params_joint)
        mean_joint_low, max_joint_low = model_low.get_error(q_valid_low, tcp_valid_low, solved_params_joint)
                
        print("\n=== Double Calibration Evaluation ===")
        print_err("JointCalib on High (Validation)", mean_joint_high, max_joint_high)
        print_err("JointCalib on Low  (Validation)", mean_joint_low, max_joint_low)
        
        stride = 6 + num_comp_param
        for i in range(0, len(solved_params_joint), stride):
            line = solved_params_joint[i:i+stride]
            print("[" + " ".join(f"{x:.4g}" for x in line) + "]", end=" \n")

    
    if compliance_set == 'Quad' or compliance_set == 'Cubic':

        compliance_model = 'Quad'
        
        num_comp_param = order_map[compliance_model]
        model_high = CalibrationModel6RComplNl(
            kinvec=KINVEC, tcp=TCP, load_fct=wrench_fct_high, comp_model=compliance_model
        )
        model_low = CalibrationModel6RComplNl(
            kinvec=KINVEC, tcp=TCP, load_fct=wrench_fct_low, comp_model=compliance_model
        )

        # ---------  para lock ----------
        
        num_params_per_ax = 6 + num_comp_param   # joint 6 kin and compliance
        

        # —— each joint ——

        sw_ax1 = make_switch([False, False, False, False, False, False], keep_comp=False, compliance_model=compliance_model)
        sw_ax2 = make_switch([True, False, False, True, True, False], keep_comp=True, compliance_model=compliance_model)
        sw_ax3 = make_switch([False, False, True, True, True, True], keep_comp=True,compliance_model=compliance_model)
        sw_ax4 = make_switch([False, True, True, True, False, True], keep_comp=True,compliance_model=compliance_model)
        sw_ax5 = make_switch([True, False, True, False, True, True], keep_comp=True,compliance_model=compliance_model)
        sw_ax6 = make_switch([False, True, True, True, False, False], keep_comp=False,compliance_model=compliance_model)
        sw_tool = make_switch([True, False, True, False, False, False], keep_comp=False,compliance_model=compliance_model)


        err_par_switch = [
            sw_ax1, sw_ax2, sw_ax3, sw_ax4, sw_ax5, sw_ax6, sw_tool
        ]

        

        model_high.set_error_par_switch(err_par_switch)
        model_low.set_error_par_switch(err_par_switch)

        # --------- initial reeor para ---------------
        init_guess = np.zeros((7 * num_params_per_ax,))

        padded = []
        for i in range(7):
            for j in range(num_params_per_ax - 1):
                padded.append(solved_params_joint[i * (num_params_per_ax - 1) + j])
            padded.append(0.0)
        solved_para = np.array(padded)


        solver_double = CalibrationSolverIpopt()
        solved_params_joint = solver_double.solve_double_calibration(
            model_high=model_high,
            model_low=model_low,
            q_high=q_calib_high,
            meas_high=tcp_calib_high,
            q_low=q_calib_low,
            meas_low=tcp_calib_low,
            comp_model=compliance_model,
            initial_guess=solved_para,
        )    
        
        if compliance_set == 'Quad':
            mean_joint_high, max_joint_high = model_high.get_error(q_valid_high, tcp_valid_high, solved_params_joint)
            mean_joint_low, max_joint_low = model_low.get_error(q_valid_low, tcp_valid_low, solved_params_joint)
                    
            print("\n=== Double Calibration Evaluation ===")
            print_err("JointCalib on High (Validation)", mean_joint_high, max_joint_high)
            print_err("JointCalib on Low  (Validation)", mean_joint_low, max_joint_low)
            
            stride = 6 + num_comp_param
            for i in range(0, len(solved_params_joint), stride):
                line = solved_params_joint[i:i+stride]
                print("[" + " ".join(f"{x:.4g}" for x in line) + "]", end=" \n")

        
        if compliance_set == 'Cubic':

            compliance_model = 'Cubic'
            
            num_comp_param = order_map[compliance_model]
            model_high = CalibrationModel6RComplNl(
                kinvec=KINVEC, tcp=TCP, load_fct=wrench_fct_high, comp_model=compliance_model
            )
            model_low = CalibrationModel6RComplNl(
                kinvec=KINVEC, tcp=TCP, load_fct=wrench_fct_low, comp_model=compliance_model
            )

            # ---------  para lock ----------
            
            num_params_per_ax = 6 + num_comp_param   # joint 6 kin and compliance
            

            # —— each joint ——

            sw_ax1 = make_switch([False, False, False, False, False, False], keep_comp=False, compliance_model=compliance_model)
            sw_ax2 = make_switch([True, False, False, True, True, False], keep_comp=True, compliance_model=compliance_model)
            sw_ax3 = make_switch([False, False, True, True, True, True], keep_comp=True,compliance_model=compliance_model)
            sw_ax4 = make_switch([False, True, True, True, False, True], keep_comp=True,compliance_model=compliance_model)
            sw_ax5 = make_switch([True, False, True, False, True, True], keep_comp=True,compliance_model=compliance_model)
            sw_ax6 = make_switch([False, True, True, True, False, False], keep_comp=False,compliance_model=compliance_model)
            sw_tool = make_switch([True, False, True, False, False, False], keep_comp=False,compliance_model=compliance_model)


            err_par_switch = [
                sw_ax1, sw_ax2, sw_ax3, sw_ax4, sw_ax5, sw_ax6, sw_tool
            ]

        
            model_high.set_error_par_switch(err_par_switch)
            model_low.set_error_par_switch(err_par_switch)

            # --------- initial reeor para ---------------
            init_guess = np.zeros((7 * num_params_per_ax,))

            padded = []
            for i in range(7):
                for j in range(num_params_per_ax - 1):
                    padded.append(solved_params_joint[i * (num_params_per_ax - 1) + j])
                padded.append(0.0)
            solved_para = np.array(padded)


            solver_double = CalibrationSolverIpopt()
            solved_params_joint = solver_double.solve_double_calibration(
                model_high=model_high,
                model_low=model_low,
                q_high=q_calib_high,
                meas_high=tcp_calib_high,
                q_low=q_calib_low,
                meas_low=tcp_calib_low,
                comp_model=compliance_model,
                initial_guess=solved_para,
            )    
            
            
            mean_joint_high, max_joint_high = model_high.get_error(q_valid_high, tcp_valid_high, solved_params_joint)
            mean_joint_low, max_joint_low = model_low.get_error(q_valid_low, tcp_valid_low, solved_params_joint)
                    
            print("\n=== Double Calibration Evaluation ===")
            print_err("JointCalib on High (Validation)", mean_joint_high, max_joint_high)
            print_err("JointCalib on Low  (Validation)", mean_joint_low, max_joint_low)
            
            stride = 6 + num_comp_param
            for i in range(0, len(solved_params_joint), stride):
                line = solved_params_joint[i:i+stride]
                print("[" + " ".join(f"{x:.4g}" for x in line) + "]", end=" \n")        
    

    #     # -------- plt compliance curves ---------
    # tau_tr = np.max(np.abs(model_high.get_gravity_torque(q_valid_high)), axis=0)
    # fig, axarr = plt.subplots(3, 2, figsize=(10, 12))
    # plt.subplots_adjust(hspace=0.4)
    # model_high.plot_compliance(axarr, solved_params_joint, tau_tr)
    # plt.show()

    
    dq_calib_high = make_delta_q(q_calib_high, group_len=120)
    dq_calib_low  = make_delta_q(q_calib_low,  group_len=120)
    # ---------- 1. 配置 PI 模型 ----------
    
    pi_beta_all = {
        0:[0.05,0.10,0.20],
        1:[0.08,0.22,0.35],
        2:[0.08,0.22,0.35],
        3:[0.04,0.10,0.18],
        4:[0.05,0.10,0.20],
        5:[0.05,0.10,0.20],
    }
    pi_enabled = [False, True, True, True, True, False]  # 哪些关节考虑 PI（True 为启用）
    
    err_param_pre = solved_params_joint

    model_high.set_pi_config(pi_beta_all, pi_enabled)
    model_low.set_pi_config(pi_beta_all, pi_enabled)

    # ---------- 2. 初始化 PI 参数 ----------
    n_err = 7 * (6 + order_map[compliance_model])        # 误差参数数量
    n_wpi_high     = sum(len(pi_beta_all[j]) for j in range(6))   # 高载
    n_wpi_low      = n_wpi_high                                   # 低载（β 相同）

    init_err       = np.zeros(n_err)
    init_wpi_high  = np.zeros(n_wpi_high)
    init_wpi_low   = np.zeros(n_wpi_low)
    
    solver = CalibrationSolverIpopt()
    stage1 = solver.solve_double_calibration_pi(
        model_high=model_high,
        model_low =model_low,
        q_high=q_calib_high,
        dq_high=dq_calib_high,
        meas_high=tcp_calib_high,
        q_low=q_calib_low,
        dq_low=dq_calib_low,
        meas_low=tcp_calib_low,
        comp_model=compliance_model,
        initial_err=err_param_pre,
        initial_wpi_high=init_wpi_high,   # 新增
        initial_wpi_low =init_wpi_low,    # 新增
        lock_err=True
    )

    w_start_high = stage1[n_err : n_err + n_wpi_high]
    w_start_low  = stage1[n_err + n_wpi_high:]
    
    stage2 = solver.solve_double_calibration_pi(
        model_high=model_high,
        model_low=model_low,
        q_high=q_calib_high,
        dq_high=dq_calib_high,
        meas_high=tcp_calib_high,
        q_low=q_calib_low,
        dq_low=dq_calib_low,
        meas_low=tcp_calib_low,
        comp_model=compliance_model,
        initial_err=err_param_pre,
        initial_wpi_high=w_start_high,     # 用阶段1的结果初始化
        initial_wpi_low =w_start_low,
        lock_err=False
    )

    err_param     = stage2[:n_err]
    w_pi_high     = stage2[n_err : n_err + n_wpi_high]
    w_pi_low      = stage2[n_err + n_wpi_high:]
    
    # ------------------------------------------------------------
    # 1.     验证集关节角增量 Δq
    # ------------------------------------------------------------
    dq_valid_high = make_delta_q(q_valid_high)
    dq_valid_low  = make_delta_q(q_valid_low)

    # ------------------------------------------------------------
    # 2.     用各自权重做迟滞补偿
    # ------------------------------------------------------------
    w_high_DM = cs.DM(w_pi_high)   # 高载权重
    w_low_DM  = cs.DM(w_pi_low)    # 低载权重

    q_valid_high_corr = np.array([
        cs.evalf(model_high.get_corrected_q(
            cs.DM(qi), cs.DM(dqi), w_high_DM)).full().squeeze()
        for qi, dqi in zip(q_valid_high, dq_valid_high)
    ])

    q_valid_low_corr = np.array([
        cs.evalf(model_low.get_corrected_q(
            cs.DM(qi), cs.DM(dqi), w_low_DM)).full().squeeze()
        for qi, dqi in zip(q_valid_low, dq_valid_low)
    ])

    # ------------------------------------------------------------
    # 3.     统计每轴修正量 (°)  —— 高载 & 低载
    # ------------------------------------------------------------
    def print_joint_correction(tag, q_raw, q_corr):
        delta = np.abs(q_corr - q_raw) * 180/np.pi  # rad → deg
        print(f"\n--- {tag} correction (deg) ---")
        for j in range(6):
            mean_deg = delta[:, j].mean()
            max_deg  = delta[:, j].max()
            print(f"Joint {j+1}: mean = {mean_deg:.3f}°,  max = {max_deg:.3f}°")

    print_joint_correction("High-load", q_valid_high, q_valid_high_corr)
    print_joint_correction("Low-load ", q_valid_low , q_valid_low_corr)

    # ------------------------------------------------------------
    # 4.     打印两组权重
    # ------------------------------------------------------------
    print("\n=== w_pi (High load) ===")
    print_w_pi_grouped(w_high_DM, pi_beta_all, pi_enabled)

    print("\n=== w_pi (Low load)  ===")
    print_w_pi_grouped(w_low_DM, pi_beta_all, pi_enabled)

    # ------------------------------------------------------------
    # 5.     末端误差评估
    # ------------------------------------------------------------
    mean_err_high, max_err_high = model_high.get_error(
        q_valid_high_corr, tcp_valid_high, err_param)
    mean_err_low , max_err_low  = model_low .get_error(
        q_valid_low_corr , tcp_valid_low , err_param)

    print("\n=== PI + Joint Error combine optimization ===")
    print(f"[High Load]  mean = {mean_err_high*1000:.4f} mm,  max = {max_err_high*1000:.4f} mm")
    print(f"[Low  Load]  mean = {mean_err_low*1000:.4f} mm,  max = {max_err_low*1000:.4f} mm")
    
    sw_ax1 = make_switch([False, False, False, False, False, False], keep_comp=False, compliance_model=compliance_model)
    sw_ax2 = make_switch([True, False, False, True, True, False], keep_comp=True, compliance_model=compliance_model)
    sw_ax3 = make_switch([False, False, True, True, True, True], keep_comp=True,compliance_model=compliance_model)
    sw_ax4 = make_switch([False, True, True, True, False, True], keep_comp=True,compliance_model=compliance_model)
    sw_ax5 = make_switch([True, False, True, False, True, True], keep_comp=True,compliance_model=compliance_model)
    sw_ax6 = make_switch([False, True, True, True, False, False], keep_comp=False,compliance_model=compliance_model)
    sw_tool = make_switch([True, False, True, False, False, False], keep_comp=False,compliance_model=compliance_model)


    err_par_switch = [
        sw_ax1, sw_ax2, sw_ax3, sw_ax4, sw_ax5, sw_ax6, sw_tool
    ]

    

    model_high.set_error_par_switch(err_par_switch)
    model_low.set_error_par_switch(err_par_switch)
    
    # ==================================================================
    # ============================================================
    # 0. Δq 预处理
    # ============================================================
    dq_calib_high = make_delta_q(q_calib_high, group_len=120)
    dq_calib_low  = make_delta_q(q_calib_low , group_len=120)

    # ============================================================
    # 1. 配置 PI β & 使能
    # ============================================================
    pi_beta_all = {
        0:[0.05,0.10,0.20],
        1:[0.08,0.22,0.35],
        2:[0.08,0.22,0.35],
        3:[0.04,0.10,0.18],
        4:[0.05,0.10,0.20],
        5:[0.05,0.10,0.20],
    }
    pi_enabled = [False, True, True, True, True, False]

    model_high.set_pi_config(pi_beta_all, pi_enabled)
    model_low .set_pi_config(pi_beta_all, pi_enabled)
    
    err_param_pre = solved_params_joint

    # ============================================================
    # 2. 初始变量尺寸
    # ============================================================
    n_err        = 7 * (6 + order_map[compliance_model])
    n_wpi_high   = sum(len(pi_beta_all[j]) for j in range(6))
    n_wpi_low    = n_wpi_high

    init_err       = np.zeros(n_err)
    init_wpi_high  = np.zeros(n_wpi_high)
    init_wpi_low   = np.zeros(n_wpi_low)

    # ============================================================
    # 3. 两阶段求解（新函数已无 τ_* 参数）
    # ============================================================
    solver = CalibrationSolverIpopt()

    # ---------- Stage-1：先锁 err，只优化两组 w_pi ----------
    stage1 = solver.solve_joint_comp_pi(
        model_high=model_high,
        model_low =model_low,
        q_high=q_calib_high, dq_high=dq_calib_high, meas_high=tcp_calib_high,
        q_low =q_calib_low , dq_low =dq_calib_low , meas_low =tcp_calib_low ,
        comp_model      = compliance_model,
        group_len       = 120,
        initial_guess   = err_param_pre,
        initial_wpi_high= init_wpi_high,
        initial_wpi_low = init_wpi_low,
        lambda_intra=5.0, lambda_inter=2.0, lambda_abs=1.0,
        lock_err=True
    )

    w_start_high = stage1[n_err : n_err + n_wpi_high]
    w_start_low  = stage1[n_err + n_wpi_high:]

    # ---------- Stage-2：联合优化 ----------
    stage2 = solver.solve_joint_comp_pi(
        model_high=model_high,
        model_low =model_low,
        q_high=q_calib_high, dq_high=dq_calib_high, meas_high=tcp_calib_high,
        q_low =q_calib_low , dq_low =dq_calib_low , meas_low =tcp_calib_low ,
        comp_model      = compliance_model,
        group_len       = 120,
        initial_guess     = err_param_pre,
        initial_wpi_high= w_start_high,
        initial_wpi_low = w_start_low,
        lambda_intra=5.0, lambda_inter=2.0, lambda_abs=1.0,
        lock_err=False
    )

    # ============================================================
    # 4. 结果拆分
    # ============================================================
    err_param   = stage2[:n_err]
    w_pi_high   = stage2[n_err : n_err + n_wpi_high]
    w_pi_low    = stage2[n_err + n_wpi_high:]

    # ============================================================
    # 5. 之后的验证集补偿、修正量统计、权重打印、误差评估
    #    —— 与之前代码完全一致，无需再改动
    # ============================================================    
    # ------------------------------------------------------------
    # 1.     验证集关节角增量 Δq
    # ------------------------------------------------------------
    dq_valid_high = make_delta_q(q_valid_high)
    dq_valid_low  = make_delta_q(q_valid_low)

    # ------------------------------------------------------------
    # 2.     用各自权重做迟滞补偿
    # ------------------------------------------------------------
    w_high_DM = cs.DM(w_pi_high)   # 高载权重
    w_low_DM  = cs.DM(w_pi_low)    # 低载权重

    q_valid_high_corr = np.array([
        cs.evalf(model_high.get_corrected_q(
            cs.DM(qi), cs.DM(dqi), w_high_DM)).full().squeeze()
        for qi, dqi in zip(q_valid_high, dq_valid_high)
    ])

    q_valid_low_corr = np.array([
        cs.evalf(model_low.get_corrected_q(
            cs.DM(qi), cs.DM(dqi), w_low_DM)).full().squeeze()
        for qi, dqi in zip(q_valid_low, dq_valid_low)
    ])

    # ------------------------------------------------------------
    # 3.     统计每轴修正量 (°)  —— 高载 & 低载
    # ------------------------------------------------------------
    def print_joint_correction(tag, q_raw, q_corr):
        delta = np.abs(q_corr - q_raw) * 180/np.pi  # rad → deg
        print(f"\n--- {tag} correction (deg) ---")
        for j in range(6):
            mean_deg = delta[:, j].mean()
            max_deg  = delta[:, j].max()
            print(f"Joint {j+1}: mean = {mean_deg:.3f}°,  max = {max_deg:.3f}°")

    print_joint_correction("High-load", q_valid_high, q_valid_high_corr)
    print_joint_correction("Low-load ", q_valid_low , q_valid_low_corr)

    # ------------------------------------------------------------
    # 4.     打印两组权重
    # ------------------------------------------------------------
    print("\n=== w_pi (High load) ===")
    print_w_pi_grouped(w_high_DM, pi_beta_all, pi_enabled)

    print("\n=== w_pi (Low load)  ===")
    print_w_pi_grouped(w_low_DM, pi_beta_all, pi_enabled)

    # ------------------------------------------------------------
    # 5.     末端误差评估
    # ------------------------------------------------------------
    mean_err_high, max_err_high = model_high.get_error(
        q_valid_high_corr, tcp_valid_high, err_param)
    mean_err_low , max_err_low  = model_low .get_error(
        q_valid_low_corr , tcp_valid_low , err_param)

    print("\n=== PI + Joint Error combine optimization ===")
    print(f"[High Load]  mean = {mean_err_high*1000:.4f} mm,  max = {max_err_high*1000:.4f} mm")
    print(f"[Low  Load]  mean = {mean_err_low*1000:.4f} mm,  max = {max_err_low*1000:.4f} mm")
    
    
    # ============================================================
    # 验证阶段：比较 Raw / Compliance / Compliance+PI 三种结果
    # ============================================================

    # -------- 0) 预计算柔顺 Δq_compl（高载、低载）
    tau_valid_high = model_high.get_gravity_torque(q_valid_high)
    tau_valid_low  = model_low .get_gravity_torque(q_valid_low)

    # err_param 中几何(6) + 柔顺(nC) → 每轴 (6+nC)；取柔顺部分进行 get_dq_from_tau
    dq_compl_high = model_high.get_dq_from_tau(tau_valid_high, err_param)
    dq_compl_low  = model_low .get_dq_from_tau(tau_valid_low , err_param)

    # -------- 1) 原始 TCP distance
    dist_raw = np.linalg.norm(tcp_valid_high - tcp_valid_low, axis=1)
    print(f"\nRaw  distance   : mean = {dist_raw.mean()*1000:.2f} mm   "
        f"max = {dist_raw.max()*1000:.2f} mm")

    # -------- 2) 仅柔顺补偿（无几何误差 & PI）
    q_only_compl_high = q_valid_high + dq_compl_high
    q_only_compl_low  = q_valid_low  + dq_compl_low

    fk_h = model_high.get_symbolic_meas_fct()
    fk_l = model_low .get_symbolic_meas_fct()
    params_zero = np.zeros_like(err_param)          # 只想看柔顺 → 几何误差置零

    tcp_compl_high = np.vstack([fk_h(qi, params_zero).full().ravel()
                                for qi in q_only_compl_high])
    tcp_compl_low  = np.vstack([fk_l(qi, params_zero).full().ravel()
                                for qi in q_only_compl_low])

    dist_compl = np.linalg.norm(tcp_compl_high - tcp_compl_low, axis=1)
    print(f"After compliance: mean = {dist_compl.mean()*1000:.2f} mm   "
        f"max = {dist_compl.max()*1000:.2f} mm")

    # -------- 3) 几何误差 + 柔顺 + PI 迟滞全补偿
    # 先做 PI 补偿得到 Δq_pi_corr，再叠加柔顺 Δq_compl
    dq_valid_high_pi = make_delta_q(q_valid_high)      # Δq 用原函数
    dq_valid_low_pi  = make_delta_q(q_valid_low)

    w_high_DM = cs.DM(w_pi_high)
    w_low_DM  = cs.DM(w_pi_low)

    q_pi_high = np.array([
        cs.evalf(model_high.get_corrected_q(
            cs.DM(qi), cs.DM(dqi), w_high_DM)).full().squeeze()
        for qi, dqi in zip(q_valid_high, dq_valid_high_pi)
    ])

    q_pi_low = np.array([
        cs.evalf(model_low.get_corrected_q(
            cs.DM(qi), cs.DM(dqi), w_low_DM)).full().squeeze()
        for qi, dqi in zip(q_valid_low, dq_valid_low_pi)
    ])

    # 叠加柔顺量
    q_full_high = q_pi_high + dq_compl_high
    q_full_low  = q_pi_low  + dq_compl_low

    tcp_full_high = np.vstack([fk_h(qi, err_param).full().ravel()
                            for qi in q_full_high])
    tcp_full_low  = np.vstack([fk_l(qi, err_param).full().ravel()
                            for qi in q_full_low])

    dist_full = np.linalg.norm(tcp_full_high - tcp_full_low, axis=1)
    print(f"After compliance + PI + geom: "
        f"mean = {dist_full.mean()*1000:.2f} mm   "
        f"max = {dist_full.max()*1000:.2f} mm")

    # -------- 4) Residual 向量差 & 距离差
    delta_pred = tcp_full_high - tcp_full_low
    delta_meas = tcp_valid_high - tcp_valid_low
    residual   = np.linalg.norm(delta_pred - delta_meas, axis=1)

    print(f"Residual after all corrections (should → 0): "
        f"mean = {residual.mean()*1000:.2f} mm   "
        f"max = {residual.max()*1000:.2f} mm")
    


if __name__ == "__main__":
    main() 
