"""
calib_models.py   6-axis 3-D calibration model with joint compliance
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import casadi as cs
import numpy as np
import utils.utils as ut
import matplotlib.pyplot as plt
from typing import Dict    



class CalibrationModel(ABC):
    
    _symbolic_meas_fct: cs.Function
    _error_param_jacobian_fct: cs.Function
    _err_par_switch: List[List[bool]]

    def __init__(self) -> None:
        self._err_par_switch = []
        self._generate_symbolic_functions()

    @abstractmethod
    def _generate_symbolic_functions(self) -> None:
        ...



    def set_error_par_switch(self, err_par_switch: List[List[bool]]) -> None:
        self._err_par_switch = err_par_switch

    def get_error_par_switch(self) -> List[List[bool]]:
        return self._err_par_switch

    def get_symbolic_meas_fct(self) -> cs.Function:
        return self._symbolic_meas_fct

    def get_error_param_jacobian_fct(self) -> cs.Function:
        return self._error_param_jacobian_fct



class CalibrationModel6RComplNl(CalibrationModel):
    
    """
    Defines the physical and COMPLIANCE structure of the robot
    """

    def __init__(
        self,
        kinvec: List[List[float]],
        tcp: List[float],
        load_fct: cs.Function,
        comp_model: str,
        tau_tr: np.ndarray = None,
    ) -> None:
        self.kinvec = kinvec
        self.rotation_axes = [
            ut.RotationAxis.Z,
            ut.RotationAxis.Y,
            ut.RotationAxis.Y,
            ut.RotationAxis.X,
            ut.RotationAxis.Y,
            ut.RotationAxis.X,
        ]
        self.tcp = tcp
        self.load_fct = load_fct
        
        self.pi_beta: Dict[int, list] = {i: [] for i in range(6)}   # 关节-β 列表
        self.pi_enabled: list = [False] * 6                         # 哪些轴启用

        # compliance Casadi
        tau = cs.SX.sym("tau")
        if comp_model == "Lin":
            C = cs.SX.sym("C")           
            q_c = C * tau
            self.num_comp_param = 1
            self.comp_fct = cs.Function("comp", [tau, C], [q_c])
        elif comp_model == "Quad":
            C = cs.SX.sym("C", 2)       
            q_c = C[0] * tau + 1e-2 * C[1] * cs.sign(tau) * tau**2
            self.num_comp_param = 2
            self.comp_fct = cs.Function("comp", [tau, C], [q_c])
        elif comp_model == "Cubic":
            C = cs.SX.sym("C", 3)        
            q_c = (
                C[0] * tau
                + 1e-2 * C[1] * cs.sign(tau) * tau**2
                + 1e-4 * C[2] * tau**3
            )
            self.num_comp_param = 3
            self.comp_fct = cs.Function("comp", [tau, C], [q_c])
        else:
            raise ValueError("Unsupported comp_model")

        super().__init__()
        
# ================================================================
#  ①  配置 PI 模型：设置各关节 β 列表 & 开关
# ================================================================
    def set_pi_config(
        self,
        beta_dict: Dict[int, list],     # { joint_idx : [β1,β2,β3] }
        enabled_axes: list              # 长度6的 bool 列表
    ) -> None:
        """
        beta_dict     允许给 0~5 号关节分别指定 play 阈值列表
        enabled_axes  True=该轴启用 PI；False=该轴忽略 PI（权重会被锁 0）
        """
        # 若某关节未给 β，默认空列表
        self.pi_beta    = {j: beta_dict.get(j, []) for j in range(6)}
        self.pi_enabled = enabled_axes.copy()


# ================================================================
#  ②  简化版 “静态” Play 算子（无历史记忆，足够做权重标定）
# ================================================================
    @staticmethod
    def _play_static(x, beta):
        # 强制 x 是 MX 类型，避免传入 SX
        x = cs.MX(x)
        cond = cs.fabs(x) <= beta
        out1 = x
        out2 = cs.sign(x) * beta
        return cs.if_else(cond, out1, out2)

# ================================================================
#  ③  返回 PI-校正后的关节角向量 q̂
# ================================================================
    def get_corrected_q(self,
                        q_in: cs.SX,          # 6×1  绝对关节角
                        dq_in: cs.SX,         # 6×1  增量 Δq = q[k]-q[k-1]
                        w_pi: cs.SX):         # N×1  PI 权重
        """
        返回 q_corr = q_in - Σ w·play(Δq, β)
        """
        q_corr = []
        ptr = 0

        # 把 q_in / dq_in 转成 MX，避免 SX+MX 报错
        q_in_mx  = cs.vertcat(*[q_in[i]  + 0*w_pi[0] for i in range(q_in.numel())])
        dq_in_mx = cs.vertcat(*[dq_in[i] + 0*w_pi[0] for i in range(dq_in.numel())])

        for j in range(6):
            dq_j = dq_in_mx[j]          # MX 标量
            q_j  = q_in_mx [j]          # MX 标量

            if self.pi_enabled[j] and self.pi_beta[j]:
                hyster = 0
                for β in self.pi_beta[j]:
                    hyster += w_pi[ptr] * self._play_static(dq_j, β)
                    ptr += 1
                q_corr.append(q_j - hyster)
            else:
                ptr += len(self.pi_beta[j])
                q_corr.append(q_j)

        return cs.vertcat(*q_corr)       # 6×1 MX 向量


    # symbolic functions
    def _generate_symbolic_functions(self) -> None:
        q = cs.SX.sym("q", 6) 
        taul = cs.SX.sym("taul", 6)
        err_params = cs.SX.sym("err_params", (6 + self.num_comp_param) * 7)   


   
        qd = cs.SX.zeros(6, 1)
        qdd = cs.SX.zeros(6, 1)
        load_trq = -self.load_fct(q, qd, qdd)[0:6, 0:3]
        for i in range(6):
            taul[i] = load_trq[i, 1] 


        T = cs.SX.eye(4)
        for joint in range(6):
            err_joint = err_params[
                joint * (6 + self.num_comp_param) : (joint + 1) * (6 + self.num_comp_param)
            ]
            T = cs.mtimes(T, self._get_joint_trafo(q[joint], taul[joint], joint, err_joint))

   
        tool_err = err_params[-(6 + self.num_comp_param) : -(self.num_comp_param)]
        T_tool = cs.mtimes(ut.trans_mat(self.tcp), ut.trans_mat(tool_err))
        T = cs.mtimes(T, T_tool)

        tcp_pos = T[0:3, 3]  # X,Y,Z
        self._symbolic_meas_fct = cs.Function("tcp3d", [q, err_params], [tcp_pos])

        jac = cs.jacobian(tcp_pos, err_params)
        self._error_param_jacobian_fct = cs.Function("jac_err", [q, err_params], [jac])

    # get final tcp
    def _get_joint_trafo(self, q_i, tau_i, ax_no, err_joint):

        trans_nom = ut.trans_mat(self.kinvec[ax_no])
        err_trans = ut.trans_mat(err_joint[:3])


        if self.num_comp_param > 0:
            dq_c = self.comp_fct(tau_i, err_joint[6:])
        else:
            dq_c = 0.0


        rot_err1 = ut.rot_mat_x(err_joint[3])
        rot_err2 = ut.rot_mat_y(err_joint[4])
        rot_err3 = ut.rot_mat_z(err_joint[5])


        if self.rotation_axes[ax_no] == ut.RotationAxis.X:
            rot_joint = ut.rot_mat_x(q_i + dq_c)
        elif self.rotation_axes[ax_no] == ut.RotationAxis.Y:
            rot_joint = ut.rot_mat_y(q_i + dq_c)
        elif self.rotation_axes[ax_no] == ut.RotationAxis.Z:
            rot_joint = ut.rot_mat_z(q_i + dq_c)
        else:
            raise ValueError("Unknown axis type")

        return cs.mtimes([trans_nom, err_trans, rot_err1, rot_err2, rot_err3, rot_joint])


    def get_error(
        self,
        q: np.ndarray,
        meas: np.ndarray,
        calib_params: np.ndarray,
    ) -> Tuple[float, float]:
        errs = [
            np.linalg.norm(self._symbolic_meas_fct(qi, calib_params).full().ravel() - mi)
            for qi, mi in zip(q, meas)
        ]
        return float(np.mean(errs)), float(np.max(errs))


    def get_gravity_torque(self, q: np.ndarray):
        q = np.atleast_2d(q)  # (N,6)
        qd = np.zeros((6, 1))
        qdd = np.zeros((6, 1))
        taus = []
        for qi in q:
            tau = -self.load_fct(qi, qd, qdd)[0:6, 0:3]
            taus.append(tau[:, 1])
        return np.asarray(taus)


    def plot_compliance(self, axarr, calib_params: np.ndarray, tau_tr: np.ndarray):

        num_pts = 400
        taus = [np.linspace(-t, t, num_pts) for t in tau_tr]

        for j in range(6):  
            C_start = j * (6 + self.num_comp_param) + 6
            C_end   = (j + 1) * (6 + self.num_comp_param)
            C = calib_params[C_start:C_end]

           
            if np.all(np.abs(C[0]) > 1e-8):
                tau_vals = taus[j].flatten()

             
                comp_map = self.comp_fct.map(num_pts)
                dq_vals = comp_map(cs.DM(tau_vals), cs.repmat(cs.DM(C), 1, num_pts)).full().flatten()

                row, col = j % 3, j // 3
                axarr[row, col].plot(tau_vals, dq_vals)
                axarr[row, col].set_title(f"Joint {j+1}")
                axarr[row, col].set_xlabel("τ [N·m]")
                axarr[row, col].set_ylabel("Δq [rad]")
                axarr[row, col].grid(True)
            else:
                print(f"Joint {j+1} compliance parameters too small, skipped plotting.")

    def get_dq_from_tau(self, tau: np.ndarray, calib_params: np.ndarray) -> np.ndarray:

        tau = np.atleast_2d(tau)  # shape (N, 6)
        dq_all = []

        for i in range(tau.shape[0]):
            dq_i = []
            for j in range(6):

                start_idx = j * (6 + self.num_comp_param) + 6
                end_idx = (j + 1) * (6 + self.num_comp_param)
                C_j = calib_params[start_idx:end_idx]
                tau_ij = tau[i, j]
                dq_ij = float(self.comp_fct(tau_ij, C_j).full().ravel()[0])
                dq_i.append(dq_ij)
            dq_all.append(dq_i)

        return np.array(dq_all)  # shape (N, 6)
