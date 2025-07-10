import numpy as np
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor_kernel
from skimage import feature
from scipy import ndimage as ndi
from functools import partial
from skimage.morphology import binary_opening,binary_closing,disk, binary_dilation
from sklearn.cluster import KMeans


def extract_glcm_features(im_gray, distances=[3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=None, properties=['dissimilarity', 'correlation'], num_patches=16):
    """
    Extract GLCM features from a grayscale image.
    
    Number of features = len(distances) * len(angles) * len(properties)
    """
    # correct patch size so that fits into the image with 0 remainder
    PATCH_SIZE=im_gray.shape[0] // num_patches
    glcm_features = np.zeros(im_gray.shape + (len(distances) * len(angles) * len(properties),), dtype=np.float32)
    for r in range(0,im_gray.shape[0],PATCH_SIZE):
        for c in range(0,im_gray.shape[1],PATCH_SIZE):
            patch=im_gray[r:r+PATCH_SIZE,c:c+PATCH_SIZE]
            for i, d in enumerate(distances):
                for j, a in enumerate(angles):
                    glcm = graycomatrix(patch, distances=[d], angles=[a], levels=levels, symmetric=True, normed=True)
                    for k, prop in enumerate(properties):
                        prop_values = graycoprops(glcm, prop)
                        glcm_features[r:r+PATCH_SIZE,c:c+PATCH_SIZE,i*len(angles)*len(properties) + j*len(properties) + k] = prop_values
    return glcm_features

def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(
        ndi.convolve(image, np.real(kernel), mode='wrap') ** 2
        + ndi.convolve(image, np.imag(kernel), mode='wrap') ** 2
    )

def return_gabor_filter_results(img_gray, thetas=None, frequencies=None):
    """
    Returns Gabor filter results.
    
    Number of features = len(thetas) * len(frequencies)
    """
    if thetas is None:
        thetas = np.linspace(0, 1.01, 3)
    if frequencies is None:
        frequencies = (0.1, 0.2, 0.3, 0.4)
        
    results = []
    kernel_params = []
    for theta in thetas:
        theta = theta / 4.0 * np.pi
        for frequency in frequencies:
            kernel = gabor_kernel(frequency, theta=theta)
            params = f"theta={theta * 180 / np.pi},\nfrequency={frequency:.2f}"
            kernel_params.append(params)
            results.append((kernel, power(img_gray, kernel)))
    return results

def extract_sobel_features(img_gray, angles=None):
    """
    Extract Sobel edge features in specified directions.
    
    Number of features = len(angles)
    """
    if angles is None:
        angles = [135, 90, 45, 180, 0, 225, 270, 315]
        
    sobel_features = []
    for angle in angles:
        theta = np.deg2rad(angle)
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        k = np.cos(theta) * kx + np.sin(theta) * ky
        sobel = ndi.convolve(img_gray, k, mode="wrap")
        sobel_features.append(sobel)
    
    return np.stack(sobel_features, axis=-1)

def extract_features(img,
                    sigma_min=1,
                    sigma_max=16,
                    glcm_distances=[3],
                    glcm_angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                    glcm_properties=['dissimilarity', 'correlation'],
                    glcm_num_patches=16,
                    gabor_thetas=None,
                    gabor_frequencies=None,
                    sobel_angles=None,
                    use_gray=True,
                    use_rgb=True,
                    use_gabor=True,
                    use_glcm=True,
                    use_sobel=True,
                    use_multiscale=True):
    """
    Extract all features from an image with toggleable feature sets.
    
    Total features depends on which feature sets are enabled:
    - Grayscale image: 1 channel (if use_gray=True)
    - Original image: 3 channels (if use_rgb=True)
    - Gabor features: len(gabor_thetas) * len(gabor_frequencies) features (if use_gabor=True)
    - GLCM features: len(glcm_distances) * len(glcm_angles) * len(glcm_properties) features (if use_glcm=True)
    - Sobel features: len(sobel_angles) features (if use_sobel=True)
    - Multiscale features: 45 features from scikit-image (if use_multiscale=True)
    """
    feature_list = []
    
    img_gray = rgb2gray(img)
    img_gray_uint8 = (img_gray*255.).astype(np.uint8)
    
    if use_gray:
        feature_list.append(np.expand_dims(img_gray, axis=-1))
        
    if use_rgb:
        feature_list.append(img)
        
    if use_gabor:
        gabor_filter_results = return_gabor_filter_results(img_gray_uint8,
                                                         thetas=gabor_thetas,
                                                         frequencies=gabor_frequencies)
        gabor_features = np.transpose(np.stack([res[1].astype(float) for res in gabor_filter_results]),(1,2,0))
        feature_list.append(gabor_features)
    
    if use_glcm:
        glcm_features = extract_glcm_features(img_gray_uint8, 
                                            distances=glcm_distances,
                                            angles=glcm_angles,
                                            properties=glcm_properties,
                                            num_patches=glcm_num_patches)
        feature_list.append(glcm_features)
    
    if use_sobel:
        sobel_features = extract_sobel_features(img_gray_uint8, angles=sobel_angles)
        feature_list.append(sobel_features)
    
    if use_multiscale:
        features_func = partial(
            feature.multiscale_basic_features,
            intensity=True,
            edges=False,
            texture=True,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            channel_axis=-1,
        )
        multi_scale_features = features_func(img)
        feature_list.append(multi_scale_features)
    
    if len(feature_list) > 0:
        all_features = np.concatenate(feature_list, axis=2)
        return all_features
    else:
        raise ValueError("At least one feature set must be enabled")

def extract_and_organize_features(X, 
                                  use_gray=True, 
                                  use_rgb=True, 
                                  use_gabor=True, 
                                  use_glcm=True, 
                                  use_sobel=True, 
                                  use_multiscale=True,
                                  gabor_thetas=np.linspace(0, 1.01, 3),
                                  gabor_frequencies=(0.1, 0.2, 0.3, 0.4),
                                  glcm_distances=[3],
                                  glcm_angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                  glcm_properties=['dissimilarity', 'correlation'],
                                  sobel_angles=[135, 90, 45, 180, 0, 225, 270, 315],
                                  sigma_min=1,
                                  sigma_max=16):
    # Calculate feature counts
    n_gabor = len(gabor_thetas) * len(gabor_frequencies)
    n_glcm = len(glcm_distances) * len(glcm_angles) * len(glcm_properties)
    n_sobel = len(sobel_angles)
    n_multiscale = 45  # from scikit-image implementation

    # Calculate total features based on enabled features
    total_features = 0
    if use_gray:
        total_features += 1
    if use_rgb:
        total_features += 3
    if use_gabor:
        total_features += n_gabor
    if use_glcm:
        total_features += n_glcm
    if use_sobel:
        total_features += n_sobel
    if use_multiscale:
        total_features += n_multiscale

    extract_features_partial = partial(extract_features,
                                     use_gray=use_gray,
                                     use_rgb=use_rgb,
                                     use_gabor=use_gabor,
                                     use_glcm=use_glcm,
                                     use_sobel=use_sobel,
                                     use_multiscale=use_multiscale,
                                     sigma_min=sigma_min,
                                     sigma_max=sigma_max,
                                     glcm_distances=glcm_distances,
                                     glcm_angles=glcm_angles,
                                     glcm_properties=glcm_properties,
                                     gabor_thetas=gabor_thetas,
                                     gabor_frequencies=gabor_frequencies,
                                     sobel_angles=sobel_angles)

    X_features = np.vectorize(extract_features_partial, signature=f'(w,h,3)->(w,h,{total_features})')(X)

    # Initialize feature indices
    current_idx = 0
    X_features_dict = {}

    # Populate feature dictionaries based on enabled features
    if use_gray:
        X_features_dict["gray"] = X_features[...,current_idx:current_idx+1]
        current_idx += 1

    if use_rgb:
        X_features_dict["RGB"] = X_features[...,current_idx:current_idx+3]
        current_idx += 3

    if use_gabor:
        X_features_dict["gabor"] = X_features[...,current_idx:current_idx+n_gabor]
        current_idx += n_gabor

    if use_glcm:
        X_features_dict["glcm"] = X_features[...,current_idx:current_idx+n_glcm]
        current_idx += n_glcm

    if use_sobel:
        X_features_dict["sobel"] = X_features[...,current_idx:current_idx+n_sobel]
        current_idx += n_sobel

    if use_multiscale:
        X_features_dict["multiscale"] = X_features[...,current_idx:]

    return X_features, X_features_dict

def postprocess_mask(mask):
    mask=binary_opening(mask,disk(5))
    mask=binary_closing(mask,disk(8))
    return mask

# for each image, calculate KMeans clustering on top 10 features
def fit_kmeans(X, n_clusters=4):
    ndim = X.shape[-1]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X.reshape(-1, ndim))
    return kmeans.labels_.reshape(X.shape[0], X.shape[1])

def assign_intensities(kmeans_labels, gray_image, n_clusters):
    cluster_intensities = []
    for i in range(n_clusters):
        cluster_intensities.append(gray_image[kmeans_labels == i].mean())
    
    # Sort clusters by intensity (lowest to highest)
    sorted_clusters = np.argsort(cluster_intensities)
    
    # Assign masks: nucleus is lowest intensity, cytoplasm is medium, background is highest
    nucleus_mask = kmeans_labels == sorted_clusters[0]
    cytoplasm_mask = np.isin(kmeans_labels, sorted_clusters[1:-1])
    
    # Postprocess masks
    nucleus_mask = postprocess_mask(nucleus_mask)
    cytoplasm_mask = postprocess_mask(cytoplasm_mask)
    cytoplasm_mask = np.logical_and(cytoplasm_mask, ~nucleus_mask)
    
    return nucleus_mask, cytoplasm_mask

def calculate_nc_ratio(nucleus_mask, cytoplasm_mask):
    nucleus_area = np.sum(nucleus_mask)
    cytoplasm_area = np.sum(cytoplasm_mask)
    return nucleus_area / (nucleus_area + cytoplasm_area)

def fit_kmeans_across_all_images(X, n_clusters=4):
    Y_kmeans = np.vectorize(partial(fit_kmeans, n_clusters=n_clusters), signature='(w,h,3)->(w,h)')(X)
    return Y_kmeans