import SimpleITK as sitk
import numpy as np
import time

def transformer_sitk(image, transform_info):
    #print('input image shape = ', image.shape)
    #print('input transform info = ', transform_info)
    image = sitk.GetImageFromArray(image)
    reference_image = image # size before and after transformation remains unchanged
    image_transformed = sitk.Resample(image,
                                      reference_image,
                                      transform_info,
                                      sitk.sitkBSpline,
                                      0.0,
                                      image.GetPixelID())
    image_arr_transformed = sitk.GetArrayViewFromImage(image_transformed)
    image_arr_transformed = np.copy(image_arr_transformed)
    return image_arr_transformed

def registration_optimization(orig_image, tilted_image, initial_transform=None, lr=0.02):
    if initial_transform is None:
        initial_transform = registration_grid_search(orig_image, tilted_image)
    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkBSpline)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=lr,
        numberOfIterations=500,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=20,
    )
    # registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace=True)

    fixed_image = sitk.GetImageFromArray(tilted_image)
    moving_image = sitk.GetImageFromArray(orig_image)
    print('Begin optimization ...')
    final_transform = registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32)
    )
    print('Done optimization.')
    print(f"Final metric value: {registration_method.GetMetricValue()}")
    print(
        f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
    )
    return final_transform

def registration_grid_search(orig_image, tilted_image, initial_transform=None, sample_per_rot_axis=11, sample_per_translation_axis=11, rot_sample_step=None, translation_sample_step=None):
    start_time = time.time()
    # convert orig and tilted image to sitk image instances
    fixed_image = sitk.GetImageFromArray(tilted_image)
    moving_image = sitk.GetImageFromArray(orig_image)
    # set up grid search space
    if rot_sample_step is None:
        if sample_per_rot_axis==0:
            rot_sample_step = [1, 1, 1]
        else:
            rot_sample_step=[2.0*np.pi/sample_per_rot_axis, 2.0*np.pi/sample_per_rot_axis, 2.0*np.pi/sample_per_rot_axis]
    if translation_sample_step is None: 
        if sample_per_translation_axis==0:
            translation_sample_step = [1, 1, 1]
        else:
            Nz, Nx, Ny = np.shape(orig_image)
            translation_sample_step = [Nz/2./sample_per_translation_axis, Nx/2./sample_per_translation_axis, Ny/2./sample_per_translation_axis]
    # sampling grid
    number_of_steps = [sample_per_rot_axis//2, sample_per_rot_axis//2, sample_per_rot_axis//2, sample_per_translation_axis//2, sample_per_translation_axis//2, sample_per_translation_axis//2]
    # sampling step size
    scales = rot_sample_step + translation_sample_step    
    print('number of steps = ', number_of_steps)
    print('search scales =', scales)
    # set up transformation
    if initial_transform is None:
        initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                              moving_image, 
                                                              sitk.Euler3DTransform(), 
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.001)
    registration_method.SetInterpolator(sitk.sitkBSpline)
    registration_method.SetOptimizerAsExhaustive(number_of_steps)
    registration_method.SetOptimizerScales(scales)
    #Perform the registration in-place so that the initial_transform is modified.
    registration_method.SetInitialTransform(initial_transform, inPlace=True)
    print("Begin grid search ...")
    registration_method.Execute(fixed_image, moving_image)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # convert orig and tilted image to sitk image instances
    print(f"Done grid search. Time elapsed {elapsed_time:.1f} sec")
    return initial_transform


