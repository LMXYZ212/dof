"""
calib_solvers.py  ipopt solvers
"""
from typing import List
import casadi as cs
import numpy as np
from calibration.calib_models import CalibrationModel


class CalibrationSolverIpopt:


    def solve_calibration(
        self,
        model: CalibrationModel,
        q: np.ndarray,
        meas: np.ndarray,
        comp_model: str,
        initial_guess: np.ndarray,
    ) -> np.ndarray:
        err_switch = model.get_error_par_switch()
        num_ax = len(err_switch)
        num_params_per_ax = len(err_switch[0])
        # print(num_params_per_ax)
        print(num_ax)

        # ---------- IPOPT opti ----------
        opti = cs.Opti()
        x = opti.variable(num_ax * num_params_per_ax)

        # ---------- cost ----------
        cost = 0.0
        fwkin = model.get_symbolic_meas_fct()
        for qi, mi in zip(q, meas):
            pred = fwkin(qi, x)
            cost += cs.sumsqr(pred - mi)
        opti.minimize(cost)

        for i in range(num_ax):
            for j in range(num_params_per_ax):
                if not err_switch[i][j]:
                    opti.subject_to(x[i * num_params_per_ax + j] == initial_guess[i * num_params_per_ax + j])

        # Restrict the first-order derivative of the compliance curve to be greater than 0 
        # and the second-order derivative to be less than 0
        if num_params_per_ax > 6:
            sample_taus = sample_taus = [x for x in range(-100, 101, 10) if x != 0]
            
            for i in range(6):
                c0_idx = i * num_params_per_ax + 6
                if err_switch[i][6] != 0:
                  
                    opti.subject_to(x[c0_idx] >= 1e-12)
                    opti.subject_to(x[c0_idx] <= 6.45161e-05)

                    if comp_model == 'Quad' or comp_model == 'Cubic':
                        opti.subject_to(x[c0_idx+1] <= 5e-05)
                        opti.subject_to(x[c0_idx+1] >= -5e-05)

                     
                        for tau in sample_taus:
                            der_quad = x[c0_idx] + 2 * 1e-2 * abs(tau) * cs.fabs(x[c0_idx+1])
                            opti.subject_to(der_quad >= 0)

                       
                        for tau in sample_taus:
                            second_der_quad = 2 * 1e-2 * x[c0_idx+1] * cs.sign(tau)
                            opti.subject_to(second_der_quad <= 0)

                        if comp_model == 'Cubic':
                            opti.subject_to(x[c0_idx+2] <= 5e-05)
                            opti.subject_to(x[c0_idx+2] >= -5e-05)

                            for tau in sample_taus:
                                der_cubic = (
                                    x[c0_idx]
                                    + 2 * 1e-2 * abs(tau) * cs.fabs(x[c0_idx+1])
                                    + 3 * 1e-4 * (tau ** 2) * x[c0_idx+2]
                                )
                                opti.subject_to(der_cubic >= 0)

                            
                            for tau in sample_taus:
                                second_der_cubic = (
                                    2 * 1e-2 * x[c0_idx+1] * cs.sign(tau)
                                    + 6 * 1e-4 * tau * x[c0_idx+2]
                                )
                                opti.subject_to(second_der_cubic <= 0)

        
        # for i in range(num_ax * num_params_per_ax):
        #     opti.subject_to(x[i] >= -0.001)
        #     opti.subject_to(x[i] <=  0.001)
        
        # ---------- solve ----------
        
        
        ipopt_opts = {
            "max_iter":        2000,
            "tol":             1e-6,
            "acceptable_tol":  1e-5,
            "acceptable_iter": 10,
            "constr_viol_tol": 1e-6,
            "dual_inf_tol":    1e-6,
            "print_level":     1,
            "mu_strategy": "adaptive",                 
            "linear_solver":   "mumps",   
        }


        opti.solver("ipopt", {}, ipopt_opts)
        opti.set_initial(x, initial_guess)
        sol = opti.solve()
        return np.asarray(sol.value(x))
    
    
    
    def solve_double_calibration(
        self,
        model_high: CalibrationModel,
        model_low: CalibrationModel,
        q_high: np.ndarray,
        meas_high: np.ndarray,
        q_low: np.ndarray,
        meas_low: np.ndarray,
        comp_model: str,
        initial_guess: np.ndarray,
    ) -> np.ndarray:

        err_switch = model_high.get_error_par_switch()
        num_ax = len(err_switch)
        num_params_per_ax = len(err_switch[0])

        assert model_low.get_error_par_switch() == err_switch, "err_switch diff!"

        # ---------- IPOPT opti ----------
        opti = cs.Opti()
        x = opti.variable(num_ax * num_params_per_ax)

        # ---------- cost  ----------
        cost = 0.0
        fwkin_high = model_high.get_symbolic_meas_fct()
        fwkin_low = model_low.get_symbolic_meas_fct()

        for qi, mi in zip(q_high, meas_high):
            pred = fwkin_high(qi, x)
            cost += cs.sumsqr(pred - mi)

        for qi, mi in zip(q_low, meas_low):
            pred = fwkin_low(qi, x)
            cost += cs.sumsqr(pred - mi)

        opti.minimize(cost)


        for i in range(num_ax):
            for j in range(num_params_per_ax):
                if not err_switch[i][j]:
                    opti.subject_to(x[i * num_params_per_ax + j] == initial_guess[i * num_params_per_ax + j])

        # Restrict the first-order derivative of the compliance curve to be greater than 0 
        # and the second-order derivative to be less than 0
        if num_params_per_ax > 6:
            sample_taus = sample_taus = [x for x in range(-100, 101, 10) if x != 0]
            
            for i in range(6):
                c0_idx = i * num_params_per_ax + 6
                if err_switch[i][6] != 0:
                    
                    opti.subject_to(x[c0_idx] >= 1e-12)
                    opti.subject_to(x[c0_idx] <= 6.45161e-05)

                    if comp_model == 'Quad' or comp_model == 'Cubic':
                        opti.subject_to(x[c0_idx+1] <= 5e-05)
                        opti.subject_to(x[c0_idx+1] >= -5e-05)

                        
                        for tau in sample_taus:
                            der_quad = x[c0_idx] + 2 * 1e-2 * abs(tau) * cs.fabs(x[c0_idx+1])
                            opti.subject_to(der_quad >= 0)

                        
                        for tau in sample_taus:
                            second_der_quad = 2 * 1e-2 * x[c0_idx+1] * cs.sign(tau)
                            opti.subject_to(second_der_quad <= 0)

                        if comp_model == 'Cubic':
                            opti.subject_to(x[c0_idx+2] <= 5e-05)
                            opti.subject_to(x[c0_idx+2] >= -5e-05)

                            
                            for tau in sample_taus:
                                der_cubic = (
                                    x[c0_idx]
                                    + 2 * 1e-2 * abs(tau) * cs.fabs(x[c0_idx+1])
                                    + 3 * 1e-4 * (tau ** 2) * x[c0_idx+2]
                                )
                                opti.subject_to(der_cubic >= 0)

                            
                            for tau in sample_taus:
                                second_der_cubic = (
                                    2 * 1e-2 * x[c0_idx+1] * cs.sign(tau)
                                    + 6 * 1e-4 * tau * x[c0_idx+2]
                                )
                                opti.subject_to(second_der_cubic <= 0)

        # for i in range(num_ax * num_params_per_ax):
        #     opti.subject_to(x[i] >= -0.001)
        #     opti.subject_to(x[i] <=  0.001)


        # ---------- solve ----------
        ipopt_opts = {
            "max_iter":        2000,
            "tol":             1e-6,
            "acceptable_tol":  1e-5,
            "acceptable_iter": 10,
            "constr_viol_tol": 1e-6,
            "dual_inf_tol":    1e-6,
            "mu_strategy": "adaptive",
            "print_level":     1,        
            "linear_solver":   "mumps",  
        }


        opti.solver("ipopt", {}, ipopt_opts)
        opti.set_initial(x, initial_guess)
        sol = opti.solve()

        return np.asarray(sol.value(x))
    
    
    def solve_compliance_alignment(
        self,
        model_high: CalibrationModel,
        model_low:  CalibrationModel,
        q_shared:   np.ndarray,        # (N,6) 
        tau_high:   np.ndarray,        # (N,6)  
        tau_low:    np.ndarray,        # (N,6) 
        meas_high:  np.ndarray,        # (N,3)  
        meas_low:   np.ndarray,        # (N,3)
        comp_model: str,
        initial_guess: np.ndarray,
        lambda1: float = 5.0,           
        lambda2: float = 2.0           
    ) -> np.ndarray:

        err_switch = model_high.get_error_par_switch()
        num_ax = len(err_switch)
        geom_param = 6
        nC = {"Lin":1, "Quad":2, "Cubic":3}[comp_model]
        num_params_per_ax = geom_param + nC

        # ---------- CasADi ----------
        opti = cs.Opti()
        x = opti.variable(num_ax * num_params_per_ax)

        fk_high = model_high.get_symbolic_meas_fct()
        fk_low  = model_low .get_symbolic_meas_fct()

        cost = 0
        for q_i, τh, τl, tcp_m_h, tcp_m_l in zip(q_shared, tau_high, tau_low,
                                                meas_high, meas_low):
    
            dq_h, dq_l = [], []
            for j in range(6):
                base = j * num_params_per_ax + geom_param
                tau_h_j = τh[j]
                tau_l_j = τl[j]

                if nC == 1:
                    C1 = x[base]
                    dq_h.append(C1 * tau_h_j)
                    dq_l.append(C1 * tau_l_j)

                elif nC == 2:
                    C1, C2 = x[base], x[base + 1]
                    dq_h.append(C1 * tau_h_j + 1e-2 * C2 * cs.sign(tau_h_j) * tau_h_j**2)
                    dq_l.append(C1 * tau_l_j + 1e-2 * C2 * cs.sign(tau_l_j) * tau_l_j**2)

                elif nC == 3:
                    C1, C2, C3 = x[base], x[base + 1], x[base + 2]
                    dq_h.append(
                        C1 * tau_h_j
                        + 1e-2 * C2 * cs.sign(tau_h_j) * tau_h_j**2
                        + 1e-4 * C3 * tau_h_j**3
                    )
                    dq_l.append(
                        C1 * tau_l_j
                        + 1e-2 * C2 * cs.sign(tau_l_j) * tau_l_j**2
                        + 1e-4 * C3 * tau_l_j**3
                    )

            dq_h = cs.vertcat(*dq_h)
            dq_l = cs.vertcat(*dq_l)

            q_corr_h = q_i + dq_h
            q_corr_l = q_i + dq_l

            tcp_pred_h = fk_high(q_corr_h, x)
            tcp_pred_l = fk_low (q_corr_l, x)

            delta_pred  = tcp_pred_h - tcp_pred_l         
            delta_meas  = tcp_m_h  - tcp_m_l              
            pred_dist2 = cs.sumsqr(tcp_pred_h - tcp_pred_l)  
            meas_dist2 = cs.sumsqr(tcp_m_h  - tcp_m_l)   
            
            cost += lambda1 * cs.sumsqr(delta_pred - delta_meas)   
            # cost += lambda2 * cs.sumsqr(pred_dist2 - meas_dist2)
            cost += cs.sumsqr(tcp_m_h - tcp_pred_h) + cs.sumsqr(tcp_m_l - tcp_pred_l)

        opti.minimize(cost)

        
        for j in range(num_ax):
           
            for g in range(geom_param):
                if not err_switch[j][g]:
                    opti.subject_to(x[j*num_params_per_ax + g] == initial_guess[j*num_params_per_ax + g])
         
            for k in range(nC):
                local_idx = geom_param + k
                if not err_switch[j][local_idx]:
                    opti.subject_to(x[j*num_params_per_ax + local_idx] == initial_guess[j*num_params_per_ax + local_idx])

        # C1 > 0
        # for j in range(6):
        #     opti.subject_to(x[j*num_params_per_ax + geom_param] > 1e-12)

        # Restrict the first-order derivative of the compliance curve to be greater than 0 
        # and the second-order derivative to be less than 0
        if num_params_per_ax > 6:
            sample_taus = sample_taus = [x for x in range(-150, 151, 5) if x != 0]
            
            for i in range(6):
                c0_idx = i * num_params_per_ax + 6
                if err_switch[i][6] != 0:
                    
                    opti.subject_to(x[c0_idx] >= 1e-12)
                    opti.subject_to(x[c0_idx] <= 6.45161e-05)

                    if comp_model == 'Quad' or comp_model == 'Cubic':
                        opti.subject_to(x[c0_idx+1] <= 5e-05)
                        opti.subject_to(x[c0_idx+1] >= -5e-05)

                        
                        for tau in sample_taus:
                            der_quad = x[c0_idx] + 2 * 1e-2 * abs(tau) * cs.fabs(x[c0_idx+1])
                            opti.subject_to(der_quad >= 0.000000001)

                        
                        for tau in sample_taus:
                            second_der_quad = 2 * 1e-2 * x[c0_idx+1] * cs.sign(tau)
                            opti.subject_to(second_der_quad <= 0)

                        if comp_model == 'Cubic':
                            opti.subject_to(x[c0_idx+2] <= 5e-05)
                            opti.subject_to(x[c0_idx+2] >= -5e-05)

                            
                            for tau in sample_taus:
                                der_cubic = (
                                    x[c0_idx]
                                    + 2 * 1e-2 * abs(tau) * cs.fabs(x[c0_idx+1])
                                    + 3 * 1e-4 * (tau ** 2) * x[c0_idx+2]
                                )
                                opti.subject_to(der_cubic >= 0)

                            
                            for tau in sample_taus:
                                second_der_cubic = (
                                    2 * 1e-2 * x[c0_idx+1] * cs.sign(tau)
                                    + 6 * 1e-4 * tau * x[c0_idx+2]
                                )
                                opti.subject_to(second_der_cubic <= 0)

        # ---------- solve ----------
        
        # ---------------- IPOPT ----------------
        ipopt_opts = {
            "max_iter":        20000,
            "tol":             1e-6,
            "acceptable_tol":  1e-5,
            "acceptable_iter": 10,
            "constr_viol_tol": 1e-6,
            "dual_inf_tol":    1e-6,
            "mu_strategy": "adaptive",
            "print_level":     1,         
            "linear_solver":   "mumps",   
        }



        opti.solver("ipopt", {}, ipopt_opts)
        opti.set_initial(x, initial_guess)
        sol = opti.solve()
        return np.asarray(sol.value(x))
    


# ----------------------------------------------------------------
    def solve_double_calibration_pi(
            self,
            model_high: CalibrationModel,
            model_low : CalibrationModel,
            q_high, dq_high, meas_high,
            q_low , dq_low , meas_low,
            comp_model: str,
            initial_err: np.ndarray,
            initial_wpi_high: np.ndarray,
            initial_wpi_low : np.ndarray,
            lock_err: bool = False
        ):
        err_switch = model_high.get_error_par_switch()
        num_ax = len(err_switch)
        num_params_per_ax = len(err_switch[0])
        n_err = num_ax * num_params_per_ax
        n_wpi_high = sum(len(model_high.pi_beta[j]) for j in range(6))
        n_wpi_low  = sum(len(model_low.pi_beta[j]) for j in range(6))

        opti = cs.Opti()
        x = opti.variable(n_err)
        w_pi_high = opti.variable(n_wpi_high)
        w_pi_low  = opti.variable(n_wpi_low)

        fk_h = model_high.get_symbolic_meas_fct()
        fk_l = model_low .get_symbolic_meas_fct()

        cost = 0
        for qh, dqh, mh in zip(q_high, dq_high, meas_high):
            qh_hat = model_high.get_corrected_q(cs.MX(qh), cs.MX(dqh), w_pi_high)
            cost += cs.sumsqr(fk_h(qh_hat, x) - mh)

        for ql, dql, ml in zip(q_low, dq_low, meas_low):
            ql_hat = model_low.get_corrected_q(cs.MX(ql), cs.MX(dql), w_pi_low)
            cost += cs.sumsqr(fk_l(ql_hat, x) - ml)

        opti.minimize(cost)

        # 误差参数锁定逻辑不变
        if lock_err:
            opti.subject_to(x == initial_err)
        else:
            for i in range(num_ax):
                for j in range(num_params_per_ax):
                    if not err_switch[i][j]:
                        idx = i * num_params_per_ax + j
                        opti.subject_to(x[idx] == initial_err[idx])

        # 高负载 π 权重锁定
        idx = 0
        for j in range(6):
            for _ in model_high.pi_beta[j]:
                if not model_high.pi_enabled[j]:
                    opti.subject_to(w_pi_high[idx] == 0)
                idx += 1

        # 低负载 π 权重锁定
        idx = 0
        for j in range(6):
            for _ in model_low.pi_beta[j]:
                if not model_low.pi_enabled[j]:
                    opti.subject_to(w_pi_low[idx] == 0)
                idx += 1

        # 柔顺性约束不变（使用 x）
        if num_params_per_ax > 6:
            sample_taus = [x for x in range(-150, 151, 5) if x != 0]
            for i in range(6):
                c0_idx = i * num_params_per_ax + 6
                if err_switch[i][6] != 0:
                    opti.subject_to(x[c0_idx] >= 1e-12)
                    opti.subject_to(x[c0_idx] <= 6.45161e-05)

                    if comp_model in ['Quad', 'Cubic']:
                        opti.subject_to(x[c0_idx+1] <= 5e-05)
                        opti.subject_to(x[c0_idx+1] >= -5e-05)
                        for tau in sample_taus:
                            der_quad = x[c0_idx] + 2 * 1e-2 * abs(tau) * cs.fabs(x[c0_idx+1])
                            opti.subject_to(der_quad >= 1e-9)

                            second_der_quad = 2 * 1e-2 * x[c0_idx+1] * cs.sign(tau)
                            opti.subject_to(second_der_quad <= 0)

                        if comp_model == 'Cubic':
                            opti.subject_to(x[c0_idx+2] <= 5e-05)
                            opti.subject_to(x[c0_idx+2] >= -5e-05)
                            for tau in sample_taus:
                                der_cubic = (
                                    x[c0_idx]
                                    + 2 * 1e-2 * abs(tau) * cs.fabs(x[c0_idx+1])
                                    + 3 * 1e-4 * (tau ** 2) * x[c0_idx+2]
                                )
                                opti.subject_to(der_cubic >= 0)

                                second_der_cubic = (
                                    2 * 1e-2 * x[c0_idx+1] * cs.sign(tau)
                                    + 6 * 1e-4 * tau * x[c0_idx+2]
                                )
                                opti.subject_to(second_der_cubic <= 0)

        # 初始化
        opti.set_initial(x, initial_err)
        opti.set_initial(w_pi_high, initial_wpi_high)
        opti.set_initial(w_pi_low , initial_wpi_low)

        opti.solver("ipopt", {}, {
            "print_level": 0,
            "max_iter": 2000,
            "linear_solver": "mumps"
        })

        sol = opti.solve()
        return np.r_[sol.value(x), sol.value(w_pi_high), sol.value(w_pi_low)]
    
    
    def solve_joint_comp_pi(
            self,
            model_high: CalibrationModel,
            model_low : CalibrationModel,
            q_high, dq_high, meas_high, 
            q_low , dq_low , meas_low , 
            comp_model: str,
            initial_guess: np.ndarray,
            initial_wpi_high: np.ndarray,
            initial_wpi_low : np.ndarray,
            lambda_intra: float = 5.0,
            lambda_inter: float = 2.0,
            lambda_abs  : float = 1.0,
            group_len: int = 120,
            lock_err: bool = False
        ):
        """
        联合优化：几何 + 柔顺 + PI(高/低载)
        - group_len: 用于确定“同一 q 出现在不同 120 组”的窗口
        """
        # ---------- 维度 ----------
        err_switch = model_high.get_error_par_switch()
        num_ax = len(err_switch)
        geom_param = 6
        nC = {"Lin":1, "Quad":2, "Cubic":3}[comp_model]
        num_params_per_ax = geom_param + nC
        n_err       = num_ax * num_params_per_ax
        n_wpi_high  = sum(len(model_high.pi_beta[j]) for j in range(6))
        n_wpi_low   = n_wpi_high

        # ---------- Opti ----------
        opti = cs.Opti()
        x           = opti.variable(n_err)
        w_pi_high   = opti.variable(n_wpi_high)
        w_pi_low    = opti.variable(n_wpi_low)

        fk_h = model_high.get_symbolic_meas_fct()
        fk_l = model_low .get_symbolic_meas_fct()
        cost = 0

        # ------------------------------------------------------------
        # 1) 绝对定位 cost_abs  (高、低各加一次)
        # ------------------------------------------------------------
        for qh, dqh, mh in zip(q_high, dq_high, meas_high):
            qh_hat = model_high.get_corrected_q(cs.MX(qh), cs.MX(dqh), w_pi_high)
            cost  += lambda_abs * cs.sumsqr(fk_h(qh_hat, x) - mh)

        for ql, dql, ml in zip(q_low, dq_low, meas_low):
            ql_hat = model_low.get_corrected_q(cs.MX(ql), cs.MX(dql), w_pi_low)
            cost  += lambda_abs * cs.sumsqr(fk_l(ql_hat, x) - ml)

        # ------------------------------------------------------------
        # 2) 高 vs 低 差异 cost_inter
        # ------------------------------------------------------------
        for qh, dqh, ql, dql, mh, ml in zip(q_high, dq_high, q_low, dq_low, meas_high, meas_low):
            qh_hat = model_high.get_corrected_q(cs.MX(qh), cs.MX(dqh), w_pi_high)
            ql_hat = model_low .get_corrected_q(cs.MX(ql), cs.MX(dql), w_pi_low )
            tcp_h  = fk_h(qh_hat, x)
            tcp_l  = fk_l(ql_hat, x)
            delta_pred  = tcp_h - tcp_l
            delta_meas  = mh     - ml
            cost += lambda_inter * cs.sumsqr(delta_pred - delta_meas)
            cost += lambda_inter * cs.sumsqr(cs.sumsqr(delta_pred) - cs.sumsqr(delta_meas))

        # ------------------------------------------------------------
        # 3) 组内迟滞 cost_intra
        # ------------------------------------------------------------
        def add_intra_cost(q_arr, dq_arr, meas_arr, model, w_pi_var):
            nonlocal cost
            # 先离线找相同 q 的索引对（Python 侧）
            from collections import defaultdict
            bucket = defaultdict(list)
            for idx, q in enumerate(q_arr):
                bucket[tuple(q)].append(idx)

            for idx_list in bucket.values():
                if len(idx_list) < 2:
                    continue
                # 每组内部两两组合
                for ii in range(len(idx_list)):
                    for jj in range(ii+1, len(idx_list)):
                        k, m = idx_list[ii], idx_list[jj]
                        qi, dqi, mi = q_arr[k], dq_arr[k], meas_arr[k]
                        qj, dqj, mj = q_arr[m], dq_arr[m], meas_arr[m]

                        q_hat_i = model.get_corrected_q(cs.MX(qi), cs.MX(dqi), w_pi_var)
                        q_hat_j = model.get_corrected_q(cs.MX(qj), cs.MX(dqj), w_pi_var)
                        tcp_i   = fk_h(q_hat_i, x) if model is model_high else fk_l(q_hat_i, x)
                        tcp_j   = fk_h(q_hat_j, x) if model is model_high else fk_l(q_hat_j, x)

                        delta_pred = tcp_i - tcp_j
                        delta_meas = mi     - mj
                        cost += lambda_intra * cs.sumsqr(delta_pred - delta_meas)
                        cost += lambda_intra * cs.sumsqr(cs.sumsqr(delta_pred) - cs.sumsqr(delta_meas))

        # 对高载、低载各加一次组内 cost
        add_intra_cost(q_high, dq_high, meas_high, model_high, w_pi_high)
        add_intra_cost(q_low , dq_low , meas_low , model_low , w_pi_low )

        # ---------- 目标 ----------
        opti.minimize(cost)

        # ---------- 约束（几何锁定 / π 禁用轴 = 0 / 柔顺单调） ----------
        
        for j in range(num_ax):
           
            for g in range(geom_param):
                if not err_switch[j][g]:
                    opti.subject_to(x[j*num_params_per_ax + g] == initial_guess[j*num_params_per_ax + g])
         
            for k in range(nC):
                local_idx = geom_param + k
                if not err_switch[j][local_idx]:
                    opti.subject_to(x[j*num_params_per_ax + local_idx] == initial_guess[j*num_params_per_ax + local_idx])

        # C1 > 0
        # for j in range(6):
        #     opti.subject_to(x[j*num_params_per_ax + geom_param] > 1e-12)

        # Restrict the first-order derivative of the compliance curve to be greater than 0 
        # and the second-order derivative to be less than 0
        if num_params_per_ax > 6:
            sample_taus = sample_taus = [x for x in range(-150, 151, 5) if x != 0]
            
            for i in range(6):
                c0_idx = i * num_params_per_ax + 6
                if err_switch[i][6] != 0:
                    
                    opti.subject_to(x[c0_idx] >= 1e-12)
                    opti.subject_to(x[c0_idx] <= 6.45161e-05)

                    if comp_model == 'Quad' or comp_model == 'Cubic':
                        opti.subject_to(x[c0_idx+1] <= 5e-05)
                        opti.subject_to(x[c0_idx+1] >= -5e-05)

                        
                        for tau in sample_taus:
                            der_quad = x[c0_idx] + 2 * 1e-2 * abs(tau) * cs.fabs(x[c0_idx+1])
                            opti.subject_to(der_quad >= 0.000000001)

                        
                        for tau in sample_taus:
                            second_der_quad = 2 * 1e-2 * x[c0_idx+1] * cs.sign(tau)
                            opti.subject_to(second_der_quad <= 0)

                        if comp_model == 'Cubic':
                            opti.subject_to(x[c0_idx+2] <= 5e-05)
                            opti.subject_to(x[c0_idx+2] >= -5e-05)

                            
                            for tau in sample_taus:
                                der_cubic = (
                                    x[c0_idx]
                                    + 2 * 1e-2 * abs(tau) * cs.fabs(x[c0_idx+1])
                                    + 3 * 1e-4 * (tau ** 2) * x[c0_idx+2]
                                )
                                opti.subject_to(der_cubic >= 0)

                            
                            for tau in sample_taus:
                                second_der_cubic = (
                                    2 * 1e-2 * x[c0_idx+1] * cs.sign(tau)
                                    + 6 * 1e-4 * tau * x[c0_idx+2]
                                )
                                opti.subject_to(second_der_cubic <= 0)
        # ---------- 初始化 & 求解 ----------
        opti.set_initial(x, initial_guess)
        opti.set_initial(w_pi_high, initial_wpi_high)
        opti.set_initial(w_pi_low , initial_wpi_low )
        opti.solver("ipopt", {}, {"print_level":0,"max_iter":5000,"linear_solver":"mumps"})
        sol = opti.solve()
        return np.r_[sol.value(x), sol.value(w_pi_high), sol.value(w_pi_low)]
