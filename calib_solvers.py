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
                if err_switch[i][j] != 0:
                  
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
                if err_switch[i][j] != 0:
                    
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
                if err_switch[i][j] != 0:
                    
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
    


    def solve_calibration_pi(
        self,
        model: CalibrationModel,
        q: np.ndarray,
        meas: np.ndarray,
        comp_model: str,
        initial_err: np.ndarray,        # ← 原误差参数初值
        initial_wpi: np.ndarray,        # ← PI 权重初值
        lock_err: bool = False          # True=锁死误差，仅优 w_pi
    ):
        """单数据集 + PI 权重   （不改老函数）"""
        # ----------- 准备维度 -----------
        err_switch = model.get_error_par_switch()
        num_ax = len(err_switch)
        num_params_per_ax = len(err_switch[0])
        n_err = num_ax * num_params_per_ax

        # 统计 PI 维度
        n_wpi = sum(len(model.pi_beta[j]) for j in range(6))

        # ----------- IPOPT -----------
        opti = cs.Opti()
        err_param = opti.variable(n_err)
        w_pi      = opti.variable(n_wpi)

        fk = model.get_symbolic_meas_fct()
        cost = 0.0
        for qi, mi in zip(q, meas):
            q_hat = model.get_corrected_q(cs.SX(qi), w_pi)
            cost += cs.sumsqr(fk(q_hat, err_param) - mi)
        opti.minimize(cost)

        # 误差锁
        if lock_err:
            opti.subject_to(err_param == initial_err)

        else:
            for i in range(num_ax):
                for j in range(num_params_per_ax):
                    if not err_switch[i][j]:
                        idx = i*num_params_per_ax + j
                        opti.subject_to(err_param[idx] == initial_err[idx])

        # 未启用轴的 w_pi 直接锁 0
        idx = 0
        for j in range(6):
            for _ in model.pi_beta[j]:
                if not model.pi_enabled[j]:
                    opti.subject_to(w_pi[idx] == 0)
                idx += 1

        # 初值
        opti.set_initial(err_param, initial_err)
        opti.set_initial(w_pi, initial_wpi)

        opti.solver("ipopt", {}, {"print_level":0,"max_iter":2000,"linear_solver":"mumps"})
        sol = opti.solve()
        return np.r_[sol.value(err_param), sol.value(w_pi)]

# ----------------------------------------------------------------
    def solve_double_calibration_pi(
        self,
        model_high: CalibrationModel,
        model_low : CalibrationModel,
        q_high, meas_high,
        q_low , meas_low,
        comp_model: str,
        initial_err: np.ndarray,
        initial_wpi: np.ndarray,
        lock_err: bool = False
    ):
        """双数据集 + PI 权重   （不改老函数）"""
        err_switch = model_high.get_error_par_switch()
        num_ax = len(err_switch)
        num_params_per_ax = len(err_switch[0])
        n_err = num_ax * num_params_per_ax
        n_wpi = sum(len(model_high.pi_beta[j]) for j in range(6))

        opti = cs.Opti()
        err_param = opti.variable(n_err)
        w_pi      = opti.variable(n_wpi)

        fk_h = model_high.get_symbolic_meas_fct()
        fk_l = model_low .get_symbolic_meas_fct()
        cost = 0
        for qh, mh in zip(q_high, meas_high):
            qh_hat = model_high.get_corrected_q(cs.MX(qh), w_pi)
            cost += cs.sumsqr(fk_h(qh_hat, err_param) - mh)
        for ql, ml in zip(q_low, meas_low):
            ql_hat = model_low.get_corrected_q(cs.MX(ql), w_pi)
            cost += cs.sumsqr(fk_l(ql_hat, err_param) - ml)
        opti.minimize(cost)

        # 锁定逻辑与上面同
        if lock_err:
            opti.subject_to(err_param == initial_err)
        else:
            for i in range(num_ax):
                for j in range(num_params_per_ax):
                    if not err_switch[i][j]:
                        idx = i*num_params_per_ax + j
                        opti.subject_to(err_param[idx] == initial_err[idx])

        idx = 0
        for j in range(6):
            for _ in model_high.pi_beta[j]:
                if not model_high.pi_enabled[j]:
                    opti.subject_to(w_pi[idx] == 0)
                idx += 1

        opti.set_initial(err_param, initial_err)
        opti.set_initial(w_pi, initial_wpi)
        opti.solver("ipopt", {}, {"print_level":0,"max_iter":2000,"linear_solver":"mumps"})
        sol = opti.solve()
        return np.r_[sol.value(err_param), sol.value(w_pi)]