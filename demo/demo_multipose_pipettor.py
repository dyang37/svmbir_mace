import os
import numpy as np
from scipy.ndimage import rotate
import SimpleITK as sitk
import svmbir
from mace import mace3D_joint_recon
from transform_utils import *
import demo_utils
import denoiser_utils

"""
This file demonstrates the generation of a 3D Shepp-Logan phantom followed by sinogram projection and reconstruction using MBIR. 
The phantom, sinogram, and reconstruction are then displayed. 
"""

phantom = np.load('demo_data/pipettor_ds4.npy')
print('shape of phantom = ', phantom.shape)
# Simulated image parameters
num_slices, num_rows_cols, _ = phantom.shape # Assumes a square image
display_slice = num_slices//2 # Display slice at z=-0.25

# Simulated sinogram parameters
num_views = num_rows_cols//6
tilt_angle = np.pi/2 # Tilt range of +-90deg

# Reconstruction parameters
sharpness = 0.0
sharpness_mace = 1.0
max_admm_itr = 20

# Display parameters
vmin = 0
vmax = 0.04

# path to store output images
save_path = './output/multipose/pipettor_ds4/'
os.makedirs(save_path, exist_ok=True)

# Rotate phantom to the second pose
phantom_rot = rotate(phantom, 45, axes=[0,2], reshape=False, order=5)
phantom_rot = rotate(phantom_rot, 30, axes=[0,1], reshape=False, order=5)

# Generate the array of view angles
angles = np.linspace(-tilt_angle, tilt_angle, num_views, endpoint=False)
angles_list = [angles for _ in range(2)]

# Generate sinogram by projecting phantom
sino_list = [svmbir.project(phantom, angles, num_rows_cols ),
             svmbir.project(phantom_rot, angles, num_rows_cols )]

# Determine resulting number of views, slices, and channels
(num_views, num_slices, num_channels) = sino_list[0].shape

# Perform fixed resolution MBIR reconstruction for each pose
recon_pose_0 = svmbir.recon(sino_list[0], angles, sharpness=sharpness)
recon_pose_1 = svmbir.recon(sino_list[1], angles, sharpness=sharpness)

#recon_pose_0 = np.load(os.path.join(save_path, 'recon_qGGMRF_pose0.npy'))
#recon_pose_1 = np.load(os.path.join(save_path, 'recon_qGGMRF_pose1.npy'))
np.save(os.path.join(save_path, 'recon_qGGMRF_pose0.npy'), recon_pose_0)
np.save(os.path.join(save_path, 'recon_qGGMRF_pose1.npy'), recon_pose_1)
# estimate transformation params with simpleITK
estimated_transform = registration_optimization(recon_pose_0, recon_pose_1, lr=0.02)
for lr in [0.01, 0.002, 0.0004]:
    estimated_transform = registration_optimization(recon_pose_0, recon_pose_1, initial_transform=estimated_transform, lr=lr)
sitk.WriteTransform(
    estimated_transform, os.path.join(save_path, "estimated_transform.tfm")
)
estimated_transform = sitk.ReadTransform(
    os.path.join(save_path, "estimated_transform.tfm")
)
recon_pose_0_transformed = transformer_sitk(recon_pose_0, estimated_transform)

diff_transform = recon_pose_1 - recon_pose_0_transformed
demo_utils.plot_image(diff_transform[display_slice], title='diff image of estimated transformation', vmin=-vmax, vmax=vmax, filename=os.path.join(save_path, 'diff_transformed_image.png'))
demo_utils.plot_image(recon_pose_0_transformed[display_slice], title='pose 0 transformed with estimated transformation', vmin=vmin, vmax=vmax, filename=os.path.join(save_path, 'pose_0_transformed.png'))
demo_utils.plot_image(recon_pose_1[display_slice], title='pose 1 transformed with estimated transformation', vmin=vmin, vmax=vmax, filename=os.path.join(save_path, 'pose_1.png'))
'''
estimated_transform = sitk.ReadTransform(
    os.path.join(save_path, "estimated_transform.tfm")
)
'''

denoiser_path = '/depot/bouman/users/yang1467/dncnn-video/models/model_dncnn_ct/'
denoiser_model = denoiser_utils.DenoiserCT(checkpoint_dir=denoiser_path)
def denoiser(img_noisy):
    img_noisy = np.expand_dims(img_noisy, axis=1)
    upper_range = denoiser_utils.calc_upper_range(img_noisy)
    img_noisy = img_noisy/upper_range
    testData_obj = denoiser_utils.DataLoader(img_noisy)
    img_denoised = denoiser_model.denoise(testData_obj, batch_size=64)
    img_denoised = img_denoised*upper_range
    return np.squeeze(img_denoised)

recon_joint = mace3D_joint_recon(sino_list, angles_list, 
                                 denoiser=denoiser,
                                 transform_info=[estimated_transform], inverse_transform_info=[estimated_transform.GetInverse()],
                                 gpu_server='gilbreth-g007', save_path=save_path,
                                 max_admm_itr=max_admm_itr, sharpness=sharpness_mace,
                                 init_image=recon_pose_0
                                 )

np.save(os.path.join(save_path, 'recon_joint.npy'), recon_joint)
# Compute Normalized Root Mean Squared Error
nrmse_pose0 = svmbir.phantom.nrmse(recon_pose_0, phantom)
nrmse_joint = svmbir.phantom.nrmse(recon_joint, phantom)
print('NRMSE of pose 0 recon = ', nrmse_pose0)
print('NRMSE of joint recon = ', nrmse_joint)
# display phantom
title = f'Slice {display_slice:d} of Pipettor Phantom.'
demo_utils.plot_image(phantom[display_slice], title=title, filename=os.path.join(save_path, 'phantom.png'), vmin=vmin, vmax=vmax)

# display fix resolution reconstruction
title = f'Slice {display_slice:d} of 3D joint recon with NRMSE={nrmse_joint:.3f}.'
demo_utils.plot_image(recon_joint[display_slice], title=title, filename=os.path.join(save_path, 'joint_recon.png'), vmin=vmin, vmax=vmax)

title = f'Slice {display_slice:d} of 3D recon of pose 0 with NRMSE={nrmse_pose0:.3f}.'
demo_utils.plot_image(recon_pose_0[display_slice], title=title, filename=os.path.join(save_path, 'recon_qGGMRF_pose0.png'), vmin=vmin, vmax=vmax)

input("press Enter")
