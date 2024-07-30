import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Helper Function: Multivariate Gaussian
def multivariate_gaussian(X, mean, cov):
    n_samples, n_features = X.shape
    norm_const = 1.0 / (np.power((2 * np.pi), float(n_features) / 2) * np.power(np.linalg.det(cov), 1.0 / 2))
    X_mean = X - mean
    result = np.einsum('ij, ij -> i', X_mean @ np.linalg.inv(cov), X_mean)
    return norm_const * np.exp(-0.5 * result)


# Initialize Parameters
def initialize_parameters(X, K):
    n_samples, n_features = X.shape
    pi = np.ones(K) / K
    mu = X[np.random.choice(n_samples, K, replace=False)]
    Sigma = np.array([np.eye(n_features)] * K)
    return pi, mu, Sigma


# E-step
def e_step(X, pi, mu, Sigma):
    n_samples, n_features = X.shape
    K = len(pi)
    gamma = np.zeros((n_samples, K))

    for k in range(K):
        gamma[:, k] = pi[k] * multivariate_gaussian(X, mu[k], Sigma[k])

    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma


# M-step
def m_step(X, gamma):
    n_samples, n_features = X.shape
    K = gamma.shape[1]

    N_k = np.sum(gamma, axis=0)
    pi = N_k / n_samples
    mu = np.dot(gamma.T, X) / N_k[:, np.newaxis]
    Sigma = np.zeros((K, n_features, n_features))

    for k in range(K):
        X_mean = X - mu[k]
        Sigma[k] = np.dot(gamma[:, k] * X_mean.T, X_mean) / N_k[k]

    return pi, mu, Sigma


# Main EM Loop
def gmm_em(X, K, max_iter=100, tol=1e-6):
    pi, mu, Sigma = initialize_parameters(X, K)
    log_likelihoods = []

    for i in range(max_iter):
        gamma = e_step(X, pi, mu, Sigma)
        pi, mu, Sigma = m_step(X, gamma)

        log_likelihood = np.sum(
            np.log(np.sum([pi[k] * multivariate_gaussian(X, mu[k], Sigma[k]) for k in range(K)], axis=0)))
        log_likelihoods.append(log_likelihood)

        if i > 0 and np.abs(log_likelihood - log_likelihoods[-2]) < tol:
            break

    return pi, mu, Sigma, gamma, log_likelihoods


# Function to perform GMM segmentation on the MRI image and plot the log-likelihood
def segment_image(image_path, is_color=True, K=3, max_iter=100, tol=1e-6):
    # Load the image
    image = Image.open(image_path)

    if is_color:
        # Convert the image to numpy array
        image_array = np.array(image)
        print(f"Original image shape: {image_array.shape}")  # Debugging line

        # If the image has an alpha channel, discard it
        if image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]

        # Normalize the pixel values
        normalized_image = image_array / 255.0
        print(f"Normalized image shape: {normalized_image.shape}")  # Debugging line

        # Reshape the image to a 2D array where each pixel is a 3D vector (R, G, B)
        pixels = normalized_image.reshape(-1, 3)
        print(f"Reshaped pixels shape: {pixels.shape}")  # Debugging line
    else:
        # Convert the image to grayscale
        gray_image = image.convert('L')
        gray_image = np.array(gray_image)

        # Normalize the pixel values
        normalized_image = gray_image / 255.0
        pixels = normalized_image.reshape(-1, 1)

    # Apply GMM
    pi, mu, Sigma, gamma, log_likelihoods = gmm_em(pixels, K, max_iter, tol)
    segmented = np.argmax(gamma, axis=1)

    # Reshape the segmented image to the original shape
    if is_color:
        segmented_image = segmented.reshape(image_array.shape[:2])
    else:
        segmented_image = segmented.reshape(gray_image.shape)

    # Plot the original and segmented images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    if is_color:
        plt.imshow(image)
    else:
        plt.imshow(gray_image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented_image, cmap='viridis')

    plt.show()

    # Plot the log-likelihood curve
    plt.figure(figsize=(8, 6))
    plt.plot(log_likelihoods, marker='o')
    plt.title('Log-Likelihood Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.show()

    # Plot the negative log-likelihood curve
    negative_log_likelihoods = [-ll for ll in log_likelihoods]
    plt.figure(figsize=(8, 6))
    plt.plot(negative_log_likelihoods, marker='o')
    plt.title('Negative Log-Likelihood Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log-Likelihood')
    plt.show()

    return pi, mu, Sigma, gamma, log_likelihoods


# Path to the MRI image
image_path1 = 'MRI.jpg'
image_path2 = 'Highway.png'

# Perform segmentation
segment_image(image_path2, K=5,is_color=True,max_iter=40)
segment_image(image_path1, K=3,is_color=False,max_iter=40)
