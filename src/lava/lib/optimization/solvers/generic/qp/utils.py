import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from lava.utils.float2fixed import *

def ruiz_equilibriation(matrix, iterations):
    """Preconditioning routine used to make the first-order QP solver converge
    faster. Returns preconditioners to be used to operate on matrices of the QP.

    Args:
        matrix (np.float): Matrix that needs to be preconditioned
        iterations (int): Number of iterations that preconditioner runs for

    Returns:
        left_preconditioner (np.float): The left preconditioning matrix
        right_preconditioner (np.float): The right preconditioning matrix

    """
    m_bar = matrix
    left_preconditioner = sparse.csc_matrix(np.eye(matrix.shape[0]))
    right_preconditioner = sparse.csc_matrix(np.eye(matrix.shape[1]))
    for _ in range(iterations):
        D_l_inv = sparse.csc_matrix(
            np.diag(1 / np.sqrt(np.linalg.norm(m_bar, ord=2, axis=1)))
        )
        if m_bar.shape[0] != m_bar.shape[1]:
            D_r_inv = sparse.csc_matrix(
                np.diag(1 / np.sqrt(np.linalg.norm(m_bar, ord=2, axis=0)))
            )
        else:
            D_r_inv = D_l_inv

        m_bar = D_l_inv @ m_bar @ D_r_inv
        left_preconditioner = left_preconditioner @ D_l_inv
    return left_preconditioner, right_preconditioner


def convert_to_fp_floor(mat, man_bits, exp_bits=None, single_block_exp=True):
    """Function that returns the exponent, mantissa representation for
    floating point numbers that need to be represented on Loihi. A global expt
    is calculated for the matrices based on the max absolute value in the
    matrix. This is then used to calculate the manstissae in the matrix.

    Args:
        mat (np.float): The input floating point matrix that needs to be
        converted
        man_bits (int): number of bits that the mantissa uses for it's
        representation (Including sign)
        single_block_exp (bool): if True, quantize with a single exponent 
        for the entire input floating point matrix, treating it as a block,
        else quantize each element with its own mantissa and exponent
        (default: True)

    Returns:
        mat_fp_man (np.int): Input matrix quantized
    """
    if man_bits <= 0:
        print("Returning without rounding")
        return mat, np.float_(0)
    if single_block_exp:
        expt = np.ceil(np.log2(np.max(np.abs(mat)))) - man_bits + 1
    else:
        expt = np.ceil(np.log2((np.abs(mat)))) - man_bits + 1
        expt[expt == -np.inf] = 0
    mat_fp_man = (mat // 2.**expt)
    if exp_bits:
        expt = np.clip(expt, -2.**(exp_bits - 1), 2.**(exp_bits - 1) - 1)
    return mat_fp_man.astype(int), expt.astype(int)


def convert_to_fp_stochastic_frexp(mat, man_bits, 
                                   exp_bits=None, 
                                   single_block_exp=True):
    if man_bits <= 0:
        print("Returning without rounding")
        return mat, np.float_(0)
    
    mant, expt = np.frexp(mat)
    
    # Scale the mantissa to the desired precision
    mant_scaled = mant * (2. ** (man_bits - 1))
    
    # Separate the integer and fractional parts
    int_mat = (np.floor(mant_scaled)).astype(np.int64)
    frac_mat = mant_scaled - int_mat
    rand_mat = np.random.rand(*mat.shape)
    rand_idx = rand_mat < frac_mat
    int_mat[rand_idx] += 1

    expt = expt - man_bits + 1

    if exp_bits:
        expt = np.clip(expt, -2.**(exp_bits - 1), 2.**(exp_bits - 1) - 1)
        
    return int_mat, expt.astype(np.int_)


def pre_condition_matrices(Q, p, A, k):
    # precondition matrices for first-order methods
    (
        pre_mat_Q,
        _,
    ) = ruiz_equilibriation(Q, 5)
    Q_pre = pre_mat_Q @ Q @ pre_mat_Q
    p_pre = pre_mat_Q @ p

    (
        pre_mat_A,
        _,
    ) = ruiz_equilibriation(A, 5)
    A_pre = pre_mat_A @ A @ pre_mat_Q
    k_pre = pre_mat_A @ k

    return Q_pre, p_pre, A_pre, k_pre


def init_alpha_beta_float(mu=0.11, sigma=8.14, lamda=1.6):

    alpha_py = 2 / (mu + 2 * lamda)
    beta_py = mu / (2 * sigma)
  
    return alpha_py, beta_py


def decay_alpha_beta_pipg_float(mu=0.11, sigma=8.14, lamda=1.6,
                                k_max=1000):
    alpha_l = [2/((k+1)*mu + 2*lamda) for k in range(k_max)]
    beta_l = [(k+1)*mu/(2*sigma) for k in range(k_max)]

    return alpha_l, beta_l


def decay_alpha_beta_loihi_float(alpha_init,
                                 beta_init,
                                 alpha_decays_at_steps=None,
                                 beta_grows_at_steps=None,
                                 k_max=1000):
    
    if alpha_decays_at_steps is None:
        alpha_decays_at_steps = [50, 100, 200, 350, 550, 800]
    if beta_grows_at_steps is None:
        beta_grows_at_steps = [1, 3, 7, 15, 31, 63, 127, 255, 511, 900]

    alpha_lava_l = [alpha_init]
    beta_lava_l = [beta_init]

    for i in range(k_max):
        if i in alpha_decays_at_steps:
            alpha_init /= 2
        if i in beta_grows_at_steps:
            beta_init *= 2
        alpha_lava_l.append(alpha_init)
        beta_lava_l.append(beta_init)

    return alpha_lava_l, beta_lava_l


def run_iters_float(Q_pre,
                    p_pre,
                    A_pre,
                    k_pre,
                    obj_fn_val_list,
                    constraint_violations_list,
                    num_iter=1000):
    
    x_pre = np.zeros_like(p_pre)
    v_pre = np.zeros_like(k_pre)
    w_pre = np.zeros_like(k_pre)

    state_x_arr = np.zeros((p_pre.shape[0], num_iter))
    state_x_arr[:, 0] = x_pre.reshape((x_pre.shape[0],))

    alpha_float, beta_float = init_alpha_beta_float()

    alpha_decays_at_steps = np.array([50, 100, 200, 350, 550, 800])
    beta_grows_at_steps = np.array([1, 3, 7, 15, 31, 63, 127, 255, 511, 900])

    for i in range(num_iter):
        if i in alpha_decays_at_steps:
            alpha_float /= 2
        x_pre = x_pre - alpha_float * (
            Q_pre @ x_pre + p_pre + A_pre.T @ v_pre
            )
        state_x_arr[:, i] = x_pre.reshape((x_pre.shape[0],))        
        if i in beta_grows_at_steps:
            beta_float *= 2
        omega = beta_float * (A_pre @ x_pre - k_pre)
        w_pre += omega
        v_pre = w_pre + omega
        
        obj_fn_val_list.append(((1/2) * (np.dot(x_pre.T, np.dot(Q_pre, x_pre))) 
                        + (np.dot(p_pre.T, x_pre))).item())
        constraint_violations_list.append(np.linalg.norm(A_pre @ x_pre - k_pre))

    return state_x_arr


def solve_qp_float(Q, p, A, k, num_iter=1000):

    obj_fn_vals = [] 
    constraint_vios = [] 

    state_x_arr = run_iters_float(Q, p, A, k,
                                  obj_fn_vals,
                                  constraint_vios, 
                                  num_iter=num_iter)

    return state_x_arr, obj_fn_vals, constraint_vios


def _dend_accum(weight_mat_mant,
                weight_mat_exp, 
                incoming_vec, 
                out_prec=32,
                rounding_mode='nearest',
                prod_mode='shift'):
    if np.isscalar(weight_mat_exp):
        weight_mat_exp = np.array([weight_mat_exp])
        # out_result = weight_mat_mant @ incoming_vec
        # out_result = left_shift(out_result, weight_mat_exp).astype(np.int_)
        # out_result = ((weight_mat_mant * (2. ** weight_mat_exp)
        #               ) @ incoming_vec)  # .astype(np.int_)
    # else:
    out_result = ((weight_mat_mant * (2. ** weight_mat_exp)
                    ) @ incoming_vec)  # .astype(np.int_)
    
    if rounding_mode == 'nearest': 
        out_mant, out_expt = convert_to_fp_floor(out_result, 
                                                 out_prec,
                                                 single_block_exp=False)
    elif rounding_mode == 'stochastic':
        out_mant, out_expt = convert_to_fp_stochastic_frexp(out_result,
                                                            out_prec,
                                                            single_block_exp=False)
    elif rounding_mode =='none':
        return out_result

    if prod_mode == 'mult':
        return np.int_(out_mant) * (2. ** out_expt)
    elif prod_mode == 'shift':
        return left_shift(np.int_(out_mant), out_expt)


def run_iters_fixed(Q_mant, Q_exp,
                    p_mant, p_exp,
                    A_mant, A_exp,
                    k_mant, k_exp,
                    num_iter=1000,
                    quantize_step_size=True,
                    da_prec=32,
                    da_rounding_mode='nearest',
                    da_prod_mode='shift'):
    
    alpha_float, beta_float = init_alpha_beta_float()
    state_x_arr = np.zeros((p_mant.shape[0], num_iter // 2))

    if quantize_step_size:
        alpha_man, alpha_exp = convert_to_fp_floor(alpha_float, man_bits=9)
        beta_man, beta_exp = convert_to_fp_floor(beta_float, man_bits=11)
    else:
        print("Not quantizing the step size.")
        alpha_man, alpha_exp = alpha_float, np.float_(0)
        beta_man, beta_exp = beta_float, np.float_(0)
    

    alpha_decays_at_steps = (
       2 * np.array([50, 100, 200, 350, 550, 800, 1100, 1450, 1850, 2300])
    )
    beta_grows_at_steps = (
       2 * np.array([1, 3, 7, 15, 31, 63, 127, 255, 511, 900, 1023, 2045, 4095]) + 1
    )

    # Initial states
    init_state_x = np.zeros(p_mant.shape).astype(np.int32)
    init_state_w = np.zeros(k_mant.shape).astype(np.int32)

    state_var_x_py = init_state_x
    state_var_w_py = init_state_w 
    gamma_py = init_state_w

    for ts in range(num_iter):
        if not np.isscalar(A_exp):
            A_exp_to_send = A_exp.T
        else:
            A_exp_to_send = A_exp
        a_in_pg_1 = _dend_accum(A_mant.T, 
                                A_exp_to_send,
                                gamma_py, 
                                out_prec=da_prec, 
                                rounding_mode=da_rounding_mode,
                                prod_mode=da_prod_mode)
        
        a_in_pg_2 = _dend_accum(Q_mant,
                                Q_exp,
                                state_var_x_py,
                                out_prec=da_prec,
                                rounding_mode=da_rounding_mode,
                                prod_mode=da_prod_mode)
        
        a_in_pg = a_in_pg_1 + a_in_pg_2

        if ts % 2 == 0:  
            if ts in alpha_decays_at_steps:
                alpha_man = alpha_man / 2.
            grad_step = a_in_pg + p_mant * (2. ** p_exp)  # ).astype(int)
            tot_bias_gd = alpha_man * grad_step

            x_inter = tot_bias_gd * (2. ** alpha_exp)  # ).astype(np.int_)

            state_var_x_py = (state_var_x_py - x_inter)  # .astype(np.int_)
            state_x_arr[:, ts//2] = \
                state_var_x_py.reshape((state_var_x_py.shape[0],))
        
        a_in_pi = _dend_accum(A_mant,
                              A_exp,
                              state_var_x_py,
                              out_prec=da_prec,
                              rounding_mode=da_rounding_mode,
                              prod_mode=da_prod_mode)

        if ts % 2 == 1:
            if ts in beta_grows_at_steps:
                beta_man *= 2.
            tot_bias_pi = beta_man * (
                a_in_pi - k_mant  # np.right_shift(k_mant, -k_exp)
            )

            omega = tot_bias_pi * (2.** beta_exp)  # ).astype(np.int_)

            state_var_w_py = state_var_w_py + omega
            gamma_py = (state_var_w_py + omega)  # .astype(np.int_)
    
    return state_x_arr
    


def solve_qp_fixed(Q, p, A, k, 
                   num_iter=1000, 
                   weight_bits={'Q': 8,
                                'A': 8,
                                'p': 24,
                                'k': 24},
                   mantissa_bits={'Q': 8,
                                  'A': 8,
                                  'p': 24,
                                  'k': 24}, 
                   single_block_exp=True,
                   weight_quantization_mode='nearest',
                   quantize_step_size=True,
                   dend_accum_prec=32,
                   dend_accum_rounding_mode='nearest',
                   dend_accum_prod_mode='shift'
                  ):
    exp_bits = {'Q': None,
                'A': None,
                'p': None,
                'k': None}
    for key in weight_bits.keys():
        if weight_bits[key] > mantissa_bits[key]:
            exp_bits[key] = weight_bits[key] - mantissa_bits[key]
    if weight_quantization_mode == 'nearest':
        Q_mant, Q_exp = convert_to_fp_floor(Q, 
                                            mantissa_bits['Q'],
                                            exp_bits=exp_bits['Q'], 
                                            single_block_exp=single_block_exp)
        A_mant, A_exp = convert_to_fp_floor(A, 
                                            mantissa_bits['A'],
                                            exp_bits=exp_bits['A'], 
                                            single_block_exp=single_block_exp)
        p_mant, p_exp = convert_to_fp_floor(p, 
                                            mantissa_bits['p'], 
                                            exp_bits=exp_bits['p'],
                                            single_block_exp=single_block_exp)
    elif weight_quantization_mode == 'stochastic':
        Q_mant, Q_exp = convert_to_fp_stochastic_frexp(Q, 
                                                       mantissa_bits['Q'],
                                                       exp_bits=exp_bits['Q'], 
                                                       single_block_exp=single_block_exp)
        A_mant, A_exp = convert_to_fp_stochastic_frexp(A, 
                                                       mantissa_bits['A'], 
                                                       exp_bits=exp_bits['A'],
                                                       single_block_exp=single_block_exp)
        p_mant, p_exp = convert_to_fp_stochastic_frexp(p, 
                                                       mantissa_bits['p'], 
                                                       exp_bits=exp_bits['p'],
                                                       single_block_exp=single_block_exp)
    elif weight_quantization_mode == 'none':
        Q_mant, Q_exp = convert_to_fp_floor(Q, 
                                            man_bits=-1, 
                                            single_block_exp=single_block_exp)
        A_mant, A_exp = convert_to_fp_floor(A, 
                                            man_bits=-1, 
                                            single_block_exp=single_block_exp)
        p_mant, p_exp = convert_to_fp_floor(p, 
                                            man_bits=-1, 
                                            single_block_exp=single_block_exp)        

    k_mant, k_exp = k.astype(int), int(0)

    state_x_arr = run_iters_fixed(Q_mant, Q_exp,
                                  p_mant, p_exp,
                                  A_mant, A_exp,
                                  k_mant, k_exp,
                                  num_iter=num_iter,
                                  quantize_step_size=quantize_step_size,
                                  da_prec=dend_accum_prec,
                                  da_rounding_mode=dend_accum_rounding_mode,
                                  da_prod_mode=dend_accum_prod_mode
                                )

    constraint_viols = (np.linalg.norm(A @ state_x_arr, 2., axis=0)).tolist()
    obj_fn_vals = (((1./2.) * np.diag(state_x_arr.T @ Q @ state_x_arr)).flatten() + (p.T @ state_x_arr).flatten()).tolist()

    return state_x_arr, obj_fn_vals, constraint_viols


def solve_qp(Q, p, A, k, num_iter=1000, **quantization_params):
    """
    quantization_params (dict):
        'do_quantize': True,  # => Fixed point, else floating
        'mantissa_bits': {'Q': 8,
                          'p': 24,
                          'A': 8,
                          'k': 24}
        'single_block_exp': True  # A single expt for entire mat
        'weight_quantization_mode': 'nearest'  # or 'stochastic'
        'quantize_step_size': True  # Quantize alpha and beta
        'dend_accum_prec': 32  # Precision of dend accum after mat @ vec
        'dend_accum_rounding_mode': 'nearest'  # or 'stochastic'
        'dend_accume_prod_mode': 'shift'  # or 'mult'
    """
    Q_pre, p_pre, A_pre, k_pre = pre_condition_matrices(Q, p, A, k)

    do_quantize = quantization_params.pop('do_quantize')

    if not do_quantize:
        state_x_arr, obj_fn, constraint_vio = solve_qp_float(Q_pre, p_pre,
                                                             A_pre, k_pre,
                                                             num_iter=num_iter)
        return state_x_arr, obj_fn, constraint_vio

    state_x_arr, obj_fn, constraint_vio = solve_qp_fixed(Q_pre, p_pre,
                                                         A_pre, k_pre,
                                                         num_iter=num_iter,
                                                         **quantization_params)

    return state_x_arr, obj_fn, constraint_vio