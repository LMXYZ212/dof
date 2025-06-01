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


def main() -> None:
   
        
    compliance_model = "Cubic"       # "Lin" / "Quad" / "Cubic"
    max_samples = 120
    validation_ratio = 0.2           

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
        print(f"{tag:<10s}  mean={mean_err*1000:.2f} mm   max={max_err*1000:.2f} mm")

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

    
    # ---------- 1. 配置 PI 模型 ----------
    
    pi_beta_all = {
        0:[0.05,0.10,0.20],
        1:[0.08,0.22,0.35],
        2:[0.08,0.22,0.35],
        3:[0.04,0.10,0.18],
        4:[0.05,0.10,0.20],
        5:[0.05,0.10,0.20],
    }
    pi_enabled = [True, True, True, False, True, False]  # 哪些关节考虑 PI（True 为启用）
    
    err_param_pre = solved_params_joint

    model_high.set_pi_config(pi_beta_all, pi_enabled)
    model_low.set_pi_config(pi_beta_all, pi_enabled)

    # ---------- 2. 初始化 PI 参数 ----------
    n_err = 7 * (6 + order_map[compliance_model])        # 误差参数数量
    n_wpi = sum(len(pi_beta_all[j]) for j in range(6))   # 所有 PI 权重数

    init_err = np.zeros(n_err)
    init_wpi = np.zeros(n_wpi)

    solver = CalibrationSolverIpopt()
    stage1 = solver.solve_double_calibration_pi(
        model_high=model_high,
        model_low=model_low,
        q_high=q_calib_high,
        meas_high=tcp_calib_high,
        q_low=q_calib_low,
        meas_low=tcp_calib_low,
        comp_model=compliance_model,
        initial_err=err_param_pre,
        initial_wpi=init_wpi,
        lock_err=True     # 锁定误差，只优化 PI 权重
    )

    stage2 = solver.solve_double_calibration_pi(
        model_high=model_high,
        model_low=model_low,
        q_high=q_calib_high,
        meas_high=tcp_calib_high,
        q_low=q_calib_low,
        meas_low=tcp_calib_low,
        comp_model=compliance_model,
        initial_err=err_param_pre,           # 几何误差用之前优化的
        initial_wpi=stage1[n_err:],          # PI 权重用阶段1得到的
        lock_err=False                       # 联合优化
    )

    err_param = stage2[:n_err]
    w_pi      = stage2[n_err:]

    q_valid_high_corr = np.array([
        cs.evalf(model_high.get_corrected_q(cs.DM(qi), cs.DM(w_pi))).full().squeeze()
        for qi in q_valid_high
    ])

    q_valid_low_corr = np.array([
        cs.evalf(model_low.get_corrected_q(cs.DM(qi), cs.DM(w_pi))).full().squeeze()
        for qi in q_valid_low
    ])

    mean_err_high, max_err_high = model_high.get_error(q_valid_high_corr, tcp_valid_high, err_param)
    mean_err_low,  max_err_low  = model_low.get_error(q_valid_low_corr,  tcp_valid_low,  err_param)

    print("\n=== PI + Joint Error combine optimization ===")
    print(f"[High Load]  mean = {mean_err_high*1000:.4f} mm,  max = {max_err_high*1000:.4f} mm")
    print(f"[Low  Load]  mean = {mean_err_low*1000:.4f} mm,  max = {max_err_low*1000:.4f} mm")

if __name__ == "__main__":
    main() 