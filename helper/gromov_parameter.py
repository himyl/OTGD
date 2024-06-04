import ot
import warnings
import numpy as np

def solve_gromov(Ca, Cb, M=None, a=None, b=None, loss='L2', symmetric=None,
                 alpha=0.5, reg=None,
                 reg_type="entropy", unbalanced=None, unbalanced_type='KL',
                 n_threads=1, method=None, max_iter=None, plan_init=None, tol=None,
                 verbose=False):
    r""" Solve the discrete (Fused) Gromov-Wasserstein and return :any:`OTResult` object

    res : OTResult()    Result of the optimization problem. The information can be obtained as follows:
        - res.plan : OT plan :math:`\mathbf{T}`
        - res.potentials : OT dual potentials
        - res.value : Optimal value of the optimization problem
        - res.value_linear : Linear OT loss with the optimal OT plan
        - res.value_quad : Quadratic (GW) part of the OT loss with the optimal OT plan
    """

    # detect backend
    nx = ot.backend.get_backend(Ca, Cb, M, a, b)

    # create uniform weights if not given
    if a is None:
        a = nx.ones(Ca.shape[0], type_as=Ca) / Ca.shape[0]
    if b is None:
        b = nx.ones(Cb.shape[1], type_as=Cb) / Cb.shape[1]

    # default values for solutions
    potentials = None
    value = None
    value_linear = None
    value_quad = None
    plan = None
    status = None
    log = None

    loss_dict = {'l2': 'square_loss', 'kl': 'kl_loss'}

    if loss.lower() not in loss_dict.keys():
        raise (NotImplementedError('Not implemented GW loss="{}"'.format(loss)))
    loss_fun = loss_dict[loss.lower()]

    if reg is None or reg == 0:  # exact OT

        if unbalanced is None and unbalanced_type.lower() not in ['semirelaxed']:  # Exact balanced OT

            if M is None or alpha == 1:  # Gromov-Wasserstein problem

                # default values for solver
                if max_iter is None:
                    max_iter = 10000
                if tol is None:
                    tol = 1e-9

                value, log = ot.gromov_wasserstein2(Ca, Cb, a, b, loss_fun=loss_fun, log=True, symmetric=symmetric, max_iter=max_iter, G0=plan_init, tol_rel=tol, tol_abs=tol, verbose=verbose)

                value_quad = value
                if alpha == 1:  # set to 0 for FGW with alpha=1
                    value_linear = 0
                plan = log['T']
                potentials = (log['u'], log['v'])

            elif alpha == 0:  # Wasserstein problem

                # default values for EMD solver
                if max_iter is None:
                    max_iter = 1000000

                value_linear, log = ot.emd2(a, b, M, numItermax=max_iter, log=True, return_matrix=True,
                                         numThreads=n_threads)

                value = value_linear
                potentials = (log['u'], log['v'])
                plan = log['G']
                status = log["warning"] if log["warning"] is not None else 'Converged'
                value_quad = 0

            else:  # Fused Gromov-Wasserstein problem

                # default values for solver
                if max_iter is None:
                    max_iter = 10000
                if tol is None:
                    tol = 1e-9

                value, log = ot.gromov.fused_gromov_wasserstein2(M, Ca, Cb, a, b, loss_fun=loss_fun, alpha=alpha, log=True,
                                                       symmetric=symmetric, max_iter=max_iter, G0=plan_init,
                                                       tol_rel=tol, tol_abs=tol, verbose=verbose)

                value_linear = log['lin_loss']
                value_quad = log['quad_loss']
                plan = log['T']
                potentials = (log['u'], log['v'])

        elif unbalanced_type.lower() in ['semirelaxed']:  # Semi-relaxed  OT

            if M is None or alpha == 1:  # Semi relaxed Gromov-Wasserstein problem

                # default values for solver
                if max_iter is None:
                    max_iter = 10000
                if tol is None:
                    tol = 1e-9

                value, log = ot.gromov.semirelaxed_gromov_wasserstein2(Ca, Cb, a, loss_fun=loss_fun, log=True,
                                                             symmetric=symmetric, max_iter=max_iter, G0=plan_init,
                                                             tol_rel=tol, tol_abs=tol, verbose=verbose)

                value_quad = value
                if alpha == 1:  # set to 0 for FGW with alpha=1
                    value_linear = 0
                plan = log['T']
                # potentials = (log['u'], log['v']) TODO

            else:  # Semi relaxed Fused Gromov-Wasserstein problem

                # default values for solver
                if max_iter is None:
                    max_iter = 10000
                if tol is None:
                    tol = 1e-9

                value, log = ot.gromov.semirelaxed_fused_gromov_wasserstein2(M, Ca, Cb, a, loss_fun=loss_fun, alpha=alpha,
                                                                   log=True, symmetric=symmetric, max_iter=max_iter,
                                                                   G0=plan_init, tol_rel=tol, tol_abs=tol,
                                                                   verbose=verbose)

                value_linear = log['lin_loss']
                value_quad = log['quad_loss']
                plan = log['T']
                # potentials = (log['u'], log['v']) TODO

        elif unbalanced_type.lower() in ['partial']:  # Partial OT

            if M is None:  # Partial Gromov-Wasserstein problem

                if unbalanced > nx.sum(a) or unbalanced > nx.sum(b):
                    raise (ValueError('Partial GW mass given in reg is too large'))
                if loss.lower() != 'l2':
                    raise (NotImplementedError('Partial GW only implemented with L2 loss'))
                if symmetric is not None:
                    raise (NotImplementedError('Partial GW only implemented with symmetric=True'))

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-7

                value, log = ot.partial.partial_gromov_wasserstein2(Ca, Cb, a, b, m=unbalanced, log=True, numItermax=max_iter,
                                                         G0=plan_init, tol=tol, verbose=verbose)

                value_quad = value
                plan = log['T']
                # potentials = (log['u'], log['v']) TODO

            else:  # partial FGW

                raise (NotImplementedError('Partial FGW not implemented yet'))

        elif unbalanced_type.lower() in ['kl', 'l2']:  # unbalanced exact OT

            raise (NotImplementedError('Unbalanced_type="{}"'.format(unbalanced_type)))

        else:
            raise (NotImplementedError('Unknown unbalanced_type="{}"'.format(unbalanced_type)))


    else:  # regularized OT

        if unbalanced is None and unbalanced_type.lower() not in ['semirelaxed']:  # Balanced regularized OT

            if reg_type.lower() in ['entropy'] and (M is None or alpha == 1):  # Entropic Gromov-Wasserstein problem

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9
                if method is None:
                    method = 'PGD'

                value_quad, log = entropic_gromov_wasserstein2(Ca, Cb, a, b, epsilon=reg, loss_fun=loss_fun, log=True,
                                                               symmetric=symmetric, solver=method, max_iter=max_iter,
                                                               G0=plan_init, tol=tol, verbose=verbose)

                plan = log['T']
                value_linear = 0
                value = value_quad + reg * nx.sum(plan * nx.log(plan + 1e-16))
                # potentials = (log['log_u'], log['log_v'])  #TODO

            elif reg_type.lower() in ['entropy'] and M is not None and alpha == 0:  # Entropic Wasserstein problem

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                plan, log = ot.bregman.sinkhorn_log(a, b, M, reg=reg, numItermax=max_iter,
                                                    stopThr=tol, log=True,
                                                    verbose=verbose)

                value_linear = nx.sum(M * plan)
                value = value_linear + reg * nx.sum(plan * nx.log(plan + 1e-16))
                potentials = (log['log_u'], log['log_v'])

            elif reg_type.lower() in ['entropy'] and M is not None:  # Entropic Fused Gromov-Wasserstein problem

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9
                if method is None:
                    method = 'PGD'

                value_noreg, log = entropic_fused_gromov_wasserstein2(M, Ca, Cb, a, b, loss_fun=loss_fun, alpha=alpha,
                                                                      log=True, symmetric=symmetric, solver=method,
                                                                      max_iter=max_iter, G0=plan_init, tol=tol,
                                                                      verbose=verbose)

                value_linear = log['lin_loss']
                value_quad = log['quad_loss']
                plan = log['T']
                # potentials = (log['u'], log['v'])
                value = value_noreg + reg * nx.sum(plan * nx.log(plan + 1e-16))

            else:
                raise (NotImplementedError('Not implemented reg_type="{}"'.format(reg_type)))

        elif unbalanced_type.lower() in ['semirelaxed']:  # Semi-relaxed  OT

            if reg_type.lower() in ['entropy'] and (M is None or alpha == 1):  # Entropic Semi-relaxed Gromov-Wasserstein problem

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                value_quad, log = ot.gromov.entropic_semirelaxed_gromov_wasserstein2(Ca, Cb, a, epsilon=reg, loss_fun=loss_fun, log=True, symmetric=symmetric, max_iter=max_iter, G0=plan_init, tol_rel=tol, tol_abs=tol, verbose=verbose)

                plan = log['T']
                value_linear = 0
                value = value_quad + reg * nx.sum(plan * nx.log(plan + 1e-16))

            else:  # Entropic Semi-relaxed FGW problem

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                value_noreg, log = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein2(M, Ca, Cb, a, loss_fun=loss_fun, alpha=alpha, log=True, symmetric=symmetric, max_iter=max_iter, G0=plan_init, tol_rel=tol, tol_abs=tol, verbose=verbose)

                value_linear = log['lin_loss']
                value_quad = log['quad_loss']
                plan = log['T']
                value = value_noreg + reg * nx.sum(plan * nx.log(plan + 1e-16))

        elif unbalanced_type.lower() in ['partial']:  # Partial OT

            if M is None:  # Partial Gromov-Wasserstein problem

                if unbalanced > nx.sum(a) or unbalanced > nx.sum(b):
                    raise (ValueError('Partial GW mass given in reg is too large'))
                if loss.lower() != 'l2':
                    raise (NotImplementedError('Partial GW only implemented with L2 loss'))
                if symmetric is not None:
                    raise (NotImplementedError('Partial GW only implemented with symmetric=True'))

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-7

                value_quad, log = ot.partial.entropic_partial_gromov_wasserstein2(Ca, Cb, a, b, reg=reg, m=unbalanced, log=True, numItermax=max_iter, G0=plan_init, tol=tol, verbose=verbose)

                value_quad = value
                plan = log['T']
                # potentials = (log['u'], log['v']) TODO

            else:  # partial FGW

                raise (NotImplementedError('Partial entropic FGW not implemented yet'))


        else:  # unbalanced AND regularized OT

            raise (NotImplementedError(
                'Not implemented reg_type="{}" and unbalanced_type="{}"'.format(reg_type, unbalanced_type)))

    res = ot.utils.OTResult(potentials=potentials, value=value,
                            value_linear=value_linear, value_quad=value_quad, plan=plan, status=status, backend=nx,
                            log=log)

    return res


def entropic_gromov_wasserstein2(
        C1, C2, p=None, q=None, loss_fun='square_loss', epsilon=0.1, symmetric=None, G0=None, max_iter=1000, tol=1e-9,
        solver='PGD', warmstart=False, verbose=False, log=False, **kwargs):
    T, logv = entropic_gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon, symmetric, G0, max_iter,
                                          tol, solver, warmstart, verbose, log=True, **kwargs)
    logv['T'] = T
    if log:
        return logv['gw_dist'], logv
    else:
        return logv['gw_dist']


def entropic_gromov_wasserstein(
        C1, C2, p=None, q=None, loss_fun='square_loss', epsilon=0.1, symmetric=None, G0=None, max_iter=1000,
        tol=1e-9, solver='PGD', warmstart=False, verbose=False, log=False, **kwargs):
    if solver not in ['PGD', 'PPA']:
        raise ValueError("Unknown solver '%s'. Pick one in ['PGD', 'PPA']." % solver)

    if loss_fun not in ('square_loss', 'kl_loss'):
        raise ValueError(f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

    C1, C2 = ot.utils.list_to_array(C1, C2)
    arr = [C1, C2]
    if p is not None:
        arr.append(ot.utils.list_to_array(p))
    else:
        p = ot.utils.unif(C1.shape[0], type_as=C1)
    if q is not None:
        arr.append(ot.utils.list_to_array(q))
    else:
        q = ot.utils.unif(C2.shape[0], type_as=C2)

    if G0 is not None:
        arr.append(G0)

    nx = ot.backend.get_backend(*arr)

    if G0 is None:
        G0 = nx.outer(p, q)

    T = G0
    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun, nx)

    if symmetric is None:
        symmetric = nx.allclose(C1, C1.T, atol=1e-10) and nx.allclose(C2, C2.T, atol=1e-10)
    if not symmetric:
        constCt, hC1t, hC2t = ot.gromov.init_matrix(C1.T, C2.T, p, q, loss_fun, nx)

    cpt = 0
    err = 1

    if warmstart:
        # initialize potentials to cope with ot.sinkhorn initialization
        N1, N2 = C1.shape[0], C2.shape[0]
        mu = nx.zeros(N1, type_as=C1) - np.log(N1)
        nu = nx.zeros(N2, type_as=C2) - np.log(N2)

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Tprev = T

        # compute the gradient
        if symmetric:
            tens = ot.gromov.gwggrad(constC, hC1, hC2, T, nx)
        else:
            tens = 0.5 * (ot.gromov.gwggrad(constC, hC1, hC2, T, nx) + ot.gromov.gwggrad(constCt, hC1t, hC2t, T, nx))

        if solver == 'PPA':
            tens = tens - epsilon * nx.log(T)

        if warmstart:
            T, loginn = sinkhorn(p, q, tens, epsilon, numItermax=max_iter, method='sinkhorn', log=True,
                                 warmstart=(mu, nu), stopThr=tol, **kwargs)
            mu = epsilon * nx.log(loginn['u'])
            nu = epsilon * nx.log(loginn['v'])

        else:
            T = sinkhorn(p, q, tens, epsilon, numItermax=max_iter, stopThr=tol, **kwargs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if abs(nx.sum(T) - 1) > 1e-5:
        warnings.warn("Solver failed to produce a transport plan. You might "
                      "want to increase the regularization parameter `epsilon`.")
    if log:
        log['gw_dist'] = ot.gromov.gwloss(constC, hC1, hC2, T, nx)
        return T, log
    else:
        return T


def sinkhorn(a, b, M, reg, method='sinkhorn', numItermax=1000, stopThr=1e-9,
             verbose=False, log=False, warn=True, warmstart=None, **kwargs):
    if method.lower() == 'sinkhorn':
        return ot.bregman.sinkhorn_knopp(a, b, M, reg, numItermax=numItermax,
                                         stopThr=stopThr, verbose=verbose, log=log,
                                         warn=warn, warmstart=warmstart,
                                         **kwargs)
    elif method.lower() == 'sinkhorn_log':
        return ot.bregman.sinkhorn_log(a, b, M, reg, numItermax=numItermax,
                                       stopThr=stopThr, verbose=verbose, log=log,
                                       warn=warn, warmstart=warmstart,
                                       **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def entropic_fused_gromov_wasserstein2(
        M, C1, C2, p=None, q=None, loss_fun='square_loss', epsilon=0.1,
        symmetric=None, alpha=0.5, G0=None, max_iter=1000, tol=1e-9,
        solver='PGD', warmstart=False, verbose=False, log=False, **kwargs):
    nx = ot.backend.get_backend(M, C1, C2)

    T, logv = entropic_fused_gromov_wasserstein(
        M, C1, C2, p, q, loss_fun, epsilon, symmetric, alpha, G0, max_iter,
        tol, solver, warmstart, verbose, log=True, **kwargs)

    logv['T'] = T

    lin_term = nx.sum(T * M)
    logv['quad_loss'] = (logv['fgw_dist'] - (1 - alpha) * lin_term)
    logv['lin_loss'] = lin_term * (1 - alpha)

    if log:
        return logv['fgw_dist'], logv
    else:
        return logv['fgw_dist']


def entropic_fused_gromov_wasserstein(
        M, C1, C2, p=None, q=None, loss_fun='square_loss', epsilon=0.1,
        symmetric=None, alpha=0.5, G0=None, max_iter=1000, tol=1e-9,
        solver='PGD', warmstart=False, verbose=False, log=False, **kwargs):
    if solver not in ['PGD', 'PPA']:
        raise ValueError("Unknown solver '%s'. Pick one in ['PGD', 'PPA']." % solver)

    if loss_fun not in ('square_loss', 'kl_loss'):
        raise ValueError(f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

    M, C1, C2 = ot.utils.list_to_array(M, C1, C2)
    arr = [M, C1, C2]
    if p is not None:
        arr.append(ot.utils.list_to_array(p))
    else:
        p = ot.utils.unif(C1.shape[0], type_as=C1)
    if q is not None:
        arr.append(ot.utils.list_to_array(q))
    else:
        q = ot.utils.unif(C2.shape[0], type_as=C2)

    if G0 is not None:
        arr.append(G0)

    nx = ot.backend.get_backend(*arr)

    if G0 is None:
        G0 = nx.outer(p, q)

    T = G0
    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun, nx)
    if symmetric is None:
        symmetric = nx.allclose(C1, C1.T, atol=1e-10) and nx.allclose(C2, C2.T, atol=1e-10)
    if not symmetric:
        constCt, hC1t, hC2t = ot.gromov.init_matrix(C1.T, C2.T, p, q, loss_fun, nx)
    cpt = 0
    err = 1

    if warmstart:
        # initialize potentials to cope with ot.sinkhorn initialization
        N1, N2 = C1.shape[0], C2.shape[0]
        mu = nx.zeros(N1, type_as=C1) - np.log(N1)
        nu = nx.zeros(N2, type_as=C2) - np.log(N2)

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Tprev = T

        # compute the gradient
        if symmetric:
            tens = alpha * ot.gromov.gwggrad(constC, hC1, hC2, T, nx) + (1 - alpha) * M
        else:
            tens = (alpha * 0.5) * (
                        ot.gromov.gwggrad(constC, hC1, hC2, T, nx) + ot.gromov.gwggrad(constCt, hC1t, hC2t, T, nx)) + (
                               1 - alpha) * M

        if solver == 'PPA':
            tens = tens - epsilon * nx.log(T)

        if warmstart:
            T, loginn = sinkhorn(p, q, tens, epsilon, method='sinkhorn', log=True, warmstart=(mu, nu),
                                 numItermax=max_iter, stopThr=tol, **kwargs)
            mu = epsilon * nx.log(loginn['u'])
            nu = epsilon * nx.log(loginn['v'])

        else:
            T = sinkhorn(p, q, tens, epsilon, method='sinkhorn', numItermax=max_iter, stopThr=tol, **kwargs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all the 10th iterations
            err = nx.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if abs(nx.sum(T) - 1) > 1e-5:
        warnings.warn(
            "Solver failed to produce a transport plan. You might want to increase the regularization parameter `epsilon`.")
    if log:
        log['fgw_dist'] = (1 - alpha) * nx.sum(M * T) + alpha * ot.gromov.gwloss(constC, hC1, hC2, T, nx)
        return T, log
    else:
        return T