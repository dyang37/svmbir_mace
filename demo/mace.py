import numpy as np
import os
import time
import paramiko
import svmbir
from transform_utils import *

__svmbir_lib_path = os.path.join(os.path.expanduser('~'), '.cache', 'svmbir')

def compute_inv_permute_vector(permute_vector):
    """ Given a permutation vector, compute its inverse permutation vector s.t. an array will have the same shape after permutation and inverse permutation. 
    """
     
    inv_permute_vector = []
    for i in range(len(permute_vector)):
        # print('i = {}'.format(i))
        position_of_i = permute_vector.index(i)
        # print('position_of_i = {}'.format(position_of_i))
        inv_permute_vector.append(position_of_i)
    return tuple(inv_permute_vector)


def denoiser_wrapper(image_noisy, denoiser, denoiser_args, permute_vector, positivity=True):
    """ This is a denoiser wrapper function. Given an image volume to be denoised, the wrapper function permutes and normalizes the image, passes it to a denoiser function, and permutes and denormalizes the denoised image back.

    Args:
        image_noisy (ndarray): image volume to be denoised
        denoiser (callable): The denoiser function to be used.

            ``denoiser(x, *denoiser_args) -> ndarray``

            where ``x`` is an ndarray of the noisy image volume, and ``denoiser_args`` is a tuple of the fixed parameters needed to completely specify the denoiser function.
        denoiser_args (tuple): [Default=()] Extra arguments passed to the denoiser function.
        permute_vector (tuple): permutation on the noisy image before passing to the denoiser function.
            It contains a permutation of [0,1,..,N-1] where N is the number of axes of image_noisy. The i’th axis of the permuted array will correspond to the axis numbered axes[i] of image_noisy. If not specified, defaults to (0,1,2), which effectively does no permutation.
            An inverse permutation is performed on the denoised image to make sure that the returned image has the same shape as the input noisy image.
        positivity: positivity constraint for denoiser output.
            If True, positivity will be enforced by clipping the denoiser output to be non-negative.

    Returns:
        ndarray: denoised image with same shape and dimensionality as input image ``image_noisy``
    """
    # permute the 3D image s.t. the desired denoising dimensionality is moved to axis=0
    image_noisy = np.transpose(image_noisy, permute_vector)
    # denoise
    image_denoised = denoiser(image_noisy, *denoiser_args)
    if positivity:
        image_denoised=np.clip(image_denoised, 0, None)
    # permute the denoised image back
    inv_permute_vector = compute_inv_permute_vector(permute_vector)
    image_denoised = np.transpose(image_denoised, inv_permute_vector)
    return image_denoised


def mace3D_joint_recon(sino_list, angles_list,
                       denoiser, transform_info, inverse_transform_info, 
                       transformer=transformer_sitk, denoiser_args=(), transformer_args=(),
                       max_admm_itr=10, rho=0.5, beta=None, prior_weight=0.5,
                       init_image=None,  
                       gpu_server=None, save_path='./',
                       geometry='parallel', dist_source_detector=None, magnification=None,
                       weights=None, weight_type='unweighted',
                       num_rows=None, num_cols=None, roi_radius=None,
                       delta_channel=1.0, delta_pixel=None, center_offset=0.0,
                       sigma_y=None, snr_db=30.0, sigma_x=None, sigma_p=None, p=1.2, q=2.0, T=1.0, b_interslice=1.0,
                       sharpness=0.0, positivity=True, max_resolutions=0, stop_threshold=0.02, max_iterations=100,
                       num_threads=None, delete_temps=True, svmbir_lib_path=__svmbir_lib_path, object_name='object',
                       verbose=1): 
    
    if verbose: 
        print("initializing MACE...")

    # verbosity level for qGGMRF recon
    verbose_qGGMRF = max(0,verbose-1)     
    # get sino_list shape 
    num_poses = len(sino_list)
    (num_views, num_det_rows, num_det_channels) = np.shape(sino_list[0])
    
    # agent weight
    if beta is None:
        if isinstance(prior_weight, (list, tuple, np.ndarray)):
            assert (len(prior_weight)==3), 'Incorrect dimensionality of prior_weight array.'
            beta = [(1-sum(prior_weight))/num_poses for _ in range(num_poses)]
            for w in prior_weight:
                beta.append(w)
        else:
            beta = []
            for k in range(num_poses):
                beta.append((1-prior_weight)/num_poses)
            for k in range(num_poses,num_poses+3):
                beta.append(prior_weight/3.)
    else:
        assert (isinstance(beta, (list, tuple, np.ndarray)) and len(beta) == num_poses+3), 'Error: agent weight must be an array or list with length num_poses+3'
    assert(all(w>=0 for w in beta) and (sum(beta)-1.)<1e-5), 'Incorrect value of prior_weight given. All elements in prior_weight should be non-negative, and sum should be no greater than 1.'   
    print('calculated agent weights = ', beta)    
    # make denoiser_args an instance if necessary
    if not isinstance(denoiser_args, tuple):
        denoiser_args = (denoiser_args,) 
        # Geometry dependent settings
    if geometry == 'parallel':
        dist_source_detector = 0.0
        magnification = 1.0
    elif geometry == 'fan':
        if dist_source_detector is None or magnification is None:
            raise Exception('For fanbeam geometry, need to specify dist_source_detector and magnification')
    else:
        raise Exception('Unrecognized geometry {}'.format(geometry))
    
    if delta_pixel is None:
        delta_pixel = delta_channel/magnification
    # Calculate automatic value of sino_listgram weights
    if weights is None:
        weights = [svmbir.calc_weights(sino_list[k], weight_type) for k in range(num_poses)]
    # Calculate automatic value of sigma_y
    if sigma_y is None:
        sigma_y = [svmbir.auto_sigma_y(sino_list[k], weights[k], snr_db, delta_pixel=delta_pixel, delta_channel=delta_channel) for k in range(num_poses)]
    # Calculate automatic value of sigma_p
    if sigma_p is None:
        sigma_p = [svmbir.auto_sigma_p(sino_list[k], delta_channel, sharpness) for k in range(num_poses)]
    # Calculate automatic value of sigma_x
    if sigma_x is None:
        sigma_x = [svmbir.auto_sigma_x(sino_list[k], delta_channel, sharpness) for k in range(num_poses)]
    elif np.isscalar(sigma_x):
        sigma_x = [sigma_x for _ in range(num_poses)]
    # setup multicluster variables
    if not gpu_server is None:
        ssh=paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(gpu_server)
        ssh_command = ("source /etc/profile \n"
                       "module load anaconda/2020.11-py38 \n"
                       "module load cuda/11.2.0\n module load cudnn/cuda-11.2_8.1\n"
                       "source activate mbircone\n"
                       "cd /depot/bouman/users/yang1467/lilly_exp/diyu/experiments/\n")
        denoising_plane_list = ['xy','yz','xz']
        # timing variables to deal with npy save lagging between two servers
        loop_time = 0
        sleep_time = 2

    (_, num_slices, num_channels) = sino_list[0].shape
    if num_rows is None:
        num_rows, _ = svmbir.auto_img_size(geometry, num_channels, delta_channel, delta_pixel, magnification)
    if num_cols is None:
        _, num_cols = svmbir.auto_img_size(geometry, num_channels, delta_channel, delta_pixel, magnification)
    # compute qGGMRF recon as init image. 
    if init_image is None:
        if verbose:
            start = time.time()
            print("Computing qGGMRF reconstruction for the first pose. This will be used as MACE initialization point.") 
        init_image = svmbir.recon(sino_list[0], angles_list[0],                                   
                                  geometry=geometry, dist_source_detector=dist_source_detector, magnification=magnification,
                                  weights=weights[0],
                                  num_rows=num_rows, num_cols=num_cols, roi_radius=roi_radius,
                                  delta_channel=delta_channel, delta_pixel=delta_pixel, center_offset=center_offset,
                                  sigma_y=sigma_y[k], sigma_x=sigma_x, p=p, q=q, T=T, b_interslice=b_interslice,
                                  positivity=positivity, max_resolutions=max_resolutions, stop_threshold=stop_threshold, max_iterations=max_iterations,
                                  num_threads=num_threads, delete_temps=delete_temps, svmbir_lib_path=svmbir_lib_path, object_name=object_name,
                                  verbose=verbose_qGGMRF)
        if verbose:
            end = time.time()
            elapsed_t = end-start
            print(f"Done computing qGGMRF reconstruction. Elapsed time: {elapsed_t:.2f} sec.")
    
    if np.isscalar(init_image):
        W = np.zeros((num_poses+3, num_slices, num_rows, num_cols)) + init_image 
        X = np.zeros((num_poses+3, num_slices, num_rows, num_cols)) + init_image 
    elif np.ndim(init_image) == 3:
        W = [np.copy(init_image) for _ in range(num_poses+3)]
        X = [np.copy(init_image) for _ in range(num_poses+3)] 
    else: 
        assert (np.ndim(init_image)==4), 'Incorrect dimensionality of init_image. Should be a 3D array.'
    # For each of the num_poses forward model agents, initialize X with the corresponding qGGMRF reconstruction of each set of sino_listgrams 
    # For each of the denoiser agents, initialize X with the first image
    #################### begin ADMM iterations
    if verbose:
        print("Begin MACE ADMM iterations:")
    for itr in range(max_admm_itr):
        if verbose:
            print(f"Begin MACE iteration {itr}/{max_admm_itr}:")
            start = time.time()
        # forward model prox map agent
        X[0] = svmbir.recon(sino_list[0], angles_list[0], 
                            init_image=X[0], prox_image=W[0],
                            geometry=geometry, dist_source_detector=dist_source_detector, magnification=magnification,
                            weights=weights[0],
                            num_rows=num_rows, num_cols=num_cols, roi_radius=roi_radius,
                            delta_channel=delta_channel, delta_pixel=delta_pixel, center_offset=center_offset,
                            sigma_y=sigma_y[0], sigma_p=sigma_p[0], p=p, q=q, T=T, b_interslice=b_interslice,
                            positivity=positivity, max_resolutions=max_resolutions, stop_threshold=stop_threshold, max_iterations=max_iterations,
                            num_threads=num_threads, delete_temps=delete_temps, svmbir_lib_path=svmbir_lib_path, object_name=object_name,
                            verbose=verbose_qGGMRF)

        for k in range(1, num_poses):
            prox_init_rot = transformer(X[k], transform_info[k-1], *transformer_args)
            prox_image_rot = transformer(W[k], transform_info[k-1], *transformer_args)
            prox_output = svmbir.recon(sino_list[k], angles_list[k],
                                       init_image=prox_init_rot, prox_image=prox_image_rot,
                                       geometry=geometry, dist_source_detector=dist_source_detector, magnification=magnification,
                                       weights=weights[k],
                                       num_rows=num_rows, num_cols=num_cols, roi_radius=roi_radius,
                                       delta_channel=delta_channel, delta_pixel=delta_pixel, center_offset=center_offset,
                                       sigma_y=sigma_y[k], sigma_p=sigma_p[k], p=p, q=q, T=T, b_interslice=b_interslice,
                                       positivity=positivity, max_resolutions=max_resolutions, stop_threshold=stop_threshold, max_iterations=max_iterations,
                                       num_threads=num_threads, delete_temps=delete_temps, svmbir_lib_path=svmbir_lib_path, object_name=object_name,
                                       verbose=verbose_qGGMRF)
               
            inverse_transform = inverse_transform_info[k-1]
            X[k] = transformer_sitk(prox_output, inverse_transform, *transformer_args)
        if verbose:
            print("Done forward model proximal map estimation.")
        # prior model denoiser agents
        time_start = time.time()
        if gpu_server is None: 
            X[num_poses] = denoiser_wrapper(W[num_poses], denoiser, denoiser_args, permute_vector=(0,1,2), positivity=positivity) # shape should be after permutation (Nz,num_poses,Nx,Ny)
            # denoising in YZ plane (along X-axis)
            X[num_poses+1] = denoiser_wrapper(W[num_poses+1], denoiser, denoiser_args, permute_vector=(1,0,2), positivity=positivity) # shape should be after permutation (Nx,num_poses,Nz,Ny)
            # denoising in XZ plane (along Y-axis)
            X[num_poses+2] = denoiser_wrapper(W[num_poses+2], denoiser, denoiser_args, permute_vector=(2,0,1), positivity=positivity) # shape should be after permutation (Ny,num_poses,Nz,Nx) 
            # Done denoising in all hyperplanes
        else:
            for i in range(num_poses, num_poses+3):
                denoiser_input_path = os.path.join(os.path.abspath(save_path), f'W{i}.npy')
                denoiser_output_path = os.path.join(os.path.abspath(save_path), f'X{i}.npy')
                denoising_plane = denoising_plane_list[i-num_poses]
                np.save(denoiser_input_path, W[i])
                denoising_command = f'python denoise_3D.py -s {denoiser_input_path} -d {denoiser_output_path} -p {denoising_plane}'
                stdin,stdout,stderr=ssh.exec_command(ssh_command+denoising_command, get_pty=True)
                stdin.close()
                stderr=stderr.readlines()
                output = ""

                stdout=stdout.readlines()
                output = ""
                for line in stdout:
                    output=output+line
                if output!="":
                    print(output)
                else:
                    print("There was no std output for this command")

                for line in stderr:
                    output=output+line
                if output!="":
                    print(output)
                else:
                    print("There was no std error for this command") 
                while not os.path.exists(denoiser_output_path):
                    time.sleep(sleep_time)
                    loop_time += sleep_time
                    if loop_time % 10 == 0:
                        print(f'Already waited for {loop_time} seconds for file {denoiser_output_path}.')
                X[i] = np.load(denoiser_output_path)
                os.remove(denoiser_input_path)
                os.remove(denoiser_output_path)
        # Done denoising in all hyperplanes
        time_end = time.time()
        time_elapsed = time_end - time_start
        if verbose:
            print(f"Done denoising in all hyer-planes, elapsed time {time_elapsed:.2f} sec")
        Z = sum([beta[k]*(2*X[k]-W[k]) for k in range(num_poses+3)]) 
        for k in range(num_poses+3):
            W[k] += 2*rho*(Z-X[k])
        if verbose:
            end = time.time()
            elapsed_t = end-start
            print(f"Done MACE iteration {itr}/{max_admm_itr}. Elapsed time: {elapsed_t:.2f} sec.")
        recon = sum([beta[k]*X[k] for k in range(num_poses+3)])
    #################### end ADMM iterations
    print("Done MACE reconstruction!")
    return recon
