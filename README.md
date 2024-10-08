# Kabsch Algorithm - Point Alignment Tool

This project implements the **Kabsch algorithm** to find the optimal rotation and translation to align two sets of points. This tool allows you to upload two images—one ground view and one elevated view—and draw points on each image. It then computes the optimal transformation to align the ground view points with the elevated view points using the Kabsch algorithm.

## Table of Contents

- [Features](#features)
- [How it Works](#how-it-works)
- [Kabsch Algorithm](#kabsch-algorithm)
  - [Mathematical Overview](#mathematical-overview)
  - [Steps of the Kabsch Algorithm](#steps-of-the-kabsch-algorithm)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Contributing](#contributing)
- [License](#license)

## Features

- Upload two images: one ground view and one elevated view.
- Draw points or lines on each image to define corresponding points.
- Calculate the optimal rotation and translation using the Kabsch algorithm to align the points from both views.
- Visualize the original and transformed points on the elevated image.

## How it Works

1. **Upload Images**: You upload two images—one as the ground view and one as the elevated view.
2. **Draw Points**: Draw corresponding points or lines on both images. You can switch between drawing modes (`point` or `line`).
3. **Calculate Transformation**: Once you've drawn enough points (at least three), the Kabsch algorithm calculates the optimal rotation and translation.
4. **Visualize Results**: The app displays the transformed ground points and the elevated points on the same image for comparison.

## Kabsch Algorithm

### Mathematical Overview

The **Kabsch algorithm** finds the optimal rotation matrix $$\( R )$$ and translation vector $$\( t )$$ that minimizes the root-mean-square deviation (RMSD) between two sets of points: $$\( P )$$ (ground points) and $$\( Q )$$ (elevated points).

Given two sets of corresponding points:
- Ground points: $$\( P = \{p_1, p_2, \dots, p_n\} \)$$
- Elevated points: $$\( Q = \{q_1, q_2, \dots, q_n\} \)$$

The goal is to find $$\( R \)$$ and $$\( t \)$$ such that:

$$\[
q_i \approx R \cdot p_i + t
\]$$

The transformation minimizes the difference between the two point sets.

### Steps of the Kabsch Algorithm

1. **Centering the Points**: 
   - Calculate the centroids of both point sets:
     
     
     $$\(Cp = \frac{1}{n} \sum_{i=1}^{n} p_i)$$

     $$\(Cq = \frac{1}{n} \sum_{i=1}^{n} q_i)$$
     

   - Center the points by subtracting the centroid from each point:
     
     
     $$\(\tilde{P} = P - Cp)$$
     
     $$\(\tilde{Q} = Q - Cq)$$

2. **Covariance Matrix**:
   - Calculate the covariance matrix $$\( H \)$$:
     $$\[
     H = \tilde{P}^T \cdot \tilde{Q}
     \]$$

3. **Singular Value Decomposition (SVD)**:
   - Perform SVD on the covariance matrix \( H \):
     $$\[
     H = U \cdot S \cdot V^T
     \]$$
   - The rotation matrix $$\( R \)$$ is given by:
     $$\[
     R = V \cdot U^T
     \]$$

4. **Special Case for Reflection**:
   - If the determinant of $$\( R \)$$ is negative (i.e., reflection is detected), negate the last row of $$\( V \)$$ and recompute $$\( R \)$$.

5. **Translation Vector**:
   - Calculate the translation vector $$\( t \)$$:
     $$\[
     t = \text{centroid}_Q - R \cdot Cp
     \]$$

6. **Apply Transformation**:
   - Finally, apply the transformation to the ground points to get the transformed points:
     $$\[
     \tilde{P}_{\text{transformed}} = R \cdot P + t
     \]$$

## Installation

To run this project locally, follow the steps below:

### Prerequisites

- Python 3.11
- [Streamlit](https://streamlit.io/) for building the web application
- OpenCV for image processing
- NumPy for numerical computations
- Matplotlib for visualizations

### Install Dependencies

1. Clone the repository:
   ```bash
   https://github.com/RyoK3N/Kabsch_Algorithm
   ```bash
   cd Kabsch_Algorithm
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
4. Running the Application:
   ```bash
   streamlit run app.py
   ```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

