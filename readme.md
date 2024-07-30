# Image Segmentation using Gaussian Mixture Model and Expectation Maximization

## Project Overview

This project demonstrates the segmentation of grayscale and color images using a Gaussian Mixture Model (GMM) and the Expectation-Maximization (EM) algorithm. The project applies these advanced machine learning techniques to segment images effectively, providing a comprehensive approach to image segmentation.

### Key Features

- **Grayscale and Color Image Segmentation**: Uses GMM and EM algorithms to segment grayscale and color images.
- **Machine Learning Application**: Demonstrates the application of machine learning techniques in image processing.
- **Log-Likelihood Plotting**: Visualizes the log-likelihood and negative log-likelihood curves to show the convergence of the EM algorithm.

### Technical Description

This section provides a detailed explanation of the core functions and algorithms used in the project.

### Getting Started

To get started with the Image Segmentation project, follow these steps to clone the repository and install the required dependencies.

#### Prerequisites

Ensure you have Python installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

#### Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/JayKareliya-code/GMM-EM-model.git
    ```

2. **Create a Virtual Environment**

    It's a good practice to use a virtual environment to manage dependencies. You can create a virtual environment using the following command:

    ```bash
    python -m venv venv
    ```

3. **Activate the Virtual Environment**

    - On Windows:

      ```bash
      venv\Scripts\activate
      ```

    - On macOS/Linux:

      ```bash
      source venv/bin/activate
      ```

4. **Install Dependencies**

    Install the required dependencies using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

5. **Run the Script**

    Once the dependencies are installed, you can run the script to perform image segmentation:

    ```bash
    python ImageSegmentation.py
    ```

### Requirements

Ensure the following libraries are included in your `requirements.txt` file:
