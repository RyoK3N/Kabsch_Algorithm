import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import math
import matplotlib.pyplot as plt

st.title('Kabsch Algorithm')

# Upload Ground and Elevated Images
ground_image = st.file_uploader("Upload Ground View Image", type=["jpg", "jpeg", "png"], key="ground")
elevated_image = st.file_uploader("Upload Elevated View Image", type=["jpg", "jpeg", "png"], key="elevated")

# Select Drawing Mode
drawing_mode = st.sidebar.selectbox("Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform"))

# Helper function to extract line points
def extract_line_points(path):
    """
    Extracts numeric coordinates from the path list and groups them into (x, y) tuples.
    
    Parameters:
        path (list): The path list containing commands and coordinates.
        
    Returns:
        list: A list of (x, y) tuples representing points.
    """
    numeric_values = [val for val in path if isinstance(val, (int, float))]
    points = []
    
    # Group numeric values into (x, y) pairs
    for i in range(0, len(numeric_values), 2):
        if i + 1 < len(numeric_values):
            x = int(numeric_values[i])
            y = int(numeric_values[i + 1])
            points.append((x, y))
    
    return points

# Function to process image with canvas
def process_image(image, image_label):
    if image is not None:
        image_pil = Image.open(image).convert("RGB")
        image_np = np.array(image_pil)

        # Get original dimensions of the uploaded image
        img_width, img_height = image_pil.size

        st.subheader(f"{image_label} Image")

        # Sidebar settings specific to each image
        stroke_width = st.sidebar.slider(f"{image_label} - Stroke width:", 1, 25, 3, key=f"{image_label}_stroke_width")
        if drawing_mode == 'point':
            point_display_radius = st.sidebar.slider(f"{image_label} - Point display radius:", 1, 25, 3, key=f"{image_label}_point_radius")
        else:
            point_display_radius = 0
        stroke_color = st.sidebar.color_picker(f"{image_label} - Stroke color hex:", key=f"{image_label}_stroke_color")
        bg_color = st.sidebar.color_picker(f"{image_label} - Background color hex:", "#eee", key=f"{image_label}_bg_color")
        bg_image = image_pil
        realtime_update = st.sidebar.checkbox(f"{image_label} - Update in realtime", True, key=f"{image_label}_realtime")

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=bg_image,
            update_streamlit=realtime_update,
            width=img_width,   # Set canvas width to the image width
            height=img_height, # Set canvas height to the image height
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key=f"canvas_{image_label}",
        )

        points = []
        if canvas_result.json_data is not None:
            # Extract points from the canvas result
            for obj in canvas_result.json_data["objects"]:
                # Handle 'circle' type for 'point' mode
                if obj["type"] == "circle" and drawing_mode == 'point':
                    cx, cy = obj["left"] + obj["radius"], obj["top"] + obj["radius"]
                    points.append((int(cx), int(cy)))

                # Handle 'path' type for 'freedraw' mode
                elif obj["type"] == "path" and drawing_mode == 'freedraw':
                    path = obj.get("path", [])
                    for point in path:
                        if isinstance(point, list) and len(point) == 2:
                            try:
                                x, y = float(point[0]), float(point[1])
                                points.append((int(x), int(y)))
                            except ValueError:
                                # Skip non-numeric entries like 'M', 'L', etc.
                                continue

                # Handle 'line' type for 'line' mode
                elif obj["type"] == "line" and drawing_mode == 'line':
                    path = obj.get("path", [])
                    extracted_points = extract_line_points(path)
                    if len(extracted_points) == 2:
                        points.extend(extracted_points)
                    else:
                        st.warning(f"Invalid line detected on the {image_label} image. Please ensure you're drawing lines correctly.")


        # If in line mode, enforce that exactly two lines (four points) are drawn
        if drawing_mode == 'line':
            if len(points) != 4:
                st.warning(f"Please draw exactly 2 lines on the {image_label} image. Currently, {len(points)//2} line(s) detected.")
                return None

        if len(points) > 0:
            # Define labels (A, B, C, D, ...)
            labels = [chr(65 + i) for i in range(len(points))]  # ASCII for A=65, B=66, C=67, etc.

            image_np_copy = image_np.copy()

            # Draw points, labels, and lines
            for i, (point, label) in enumerate(zip(points, labels)):
                # Draw the label next to each point with a smaller font size
                cv2.putText(image_np_copy, label, (point[0] + 10, point[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Draw lines connecting the points based on drawing mode
            if drawing_mode == 'line':
                # Assuming lines are drawn in pairs (start and end)
                for i in range(0, len(points), 2):
                    cv2.line(image_np_copy, points[i], points[i+1], (255, 0, 0), 2)
            else:
                # For other drawing modes like point or freedraw, connect sequentially
                for i in range(1, len(points)):
                    cv2.line(image_np_copy, points[i - 1], points[i], (255, 0, 0), 1)

            # Function to calculate angle between three points (A-B-C)
            def calculate_angle(A, B, C):
                BA = np.array(A) - np.array(B)
                BC = np.array(C) - np.array(B)
                
                cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
                # Clamp the cosine to the valid range to avoid numerical issues
                cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
                angle = np.arccos(cosine_angle)
                return np.degrees(angle)

            # Ensure at least 3 points are present to calculate an angle
            if len(points) >= 3:
                # Calculate and draw angles between consecutive lines
                for i in range(1, len(points) - 1):
                    angle = calculate_angle(points[i - 1], points[i], points[i + 1])
                    angle_text = f"{angle:.1f}°"
                    
                    # Display the angle at point[i] (intersection point of AB and BC)
                    cv2.putText(image_np_copy, angle_text, (points[i][0] - 30, points[i][1] + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Display the image with labels, lines, and angles
            st.subheader(f"{image_label} Image with Annotations")
            st.image(image_np_copy, use_column_width=True)

            # Display coordinates of points with labels
            labeled_points = {label: point for label, point in zip(labels, points)}
            st.write(f"Coordinates of Points with Labels for {image_label} Image:", labeled_points)
        else:
            st.write(f"No points drawn yet on the {image_label} image.")

        return points  # Return the points for potential further processing

    return None

# Process Ground Image
ground_points = process_image(ground_image, "Ground")

# Process Elevated Image
elevated_points = process_image(elevated_image, "Elevated")

# Optional: Display Kabsch Algorithm Results if both sets of points are available
if ground_points and elevated_points:
    st.subheader("Kabsch Algorithm Results")

    # Ensure both point sets have the same number of points
    min_points = min(len(ground_points), len(elevated_points))
    if min_points < 3:
        st.warning("Please ensure at least 3 corresponding points are drawn on both images.")
    else:
        # Truncate to the minimum number of points
        ground_points_np = np.array(ground_points[:min_points])
        elevated_points_np = np.array(elevated_points[:min_points])

        # Compute the centroids
        centroid_ground = np.mean(ground_points_np, axis=0)
        centroid_elevated = np.mean(elevated_points_np, axis=0)

        # Center the points
        ground_centered = ground_points_np - centroid_ground
        elevated_centered = elevated_points_np - centroid_elevated

        # Compute the covariance matrix
        H = ground_centered.T @ elevated_centered

        # Perform Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Special reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute the translation
        t = centroid_elevated - R @ centroid_ground

        st.write("Rotation Matrix (R):")
        st.write(R)
        st.write("Translation Vector (t):")
        st.write(t)

        # Apply the transformation to ground points
        transformed_ground = (R @ ground_points_np.T).T + t

        # Display matched points
        st.write("Transformed Ground Points:")
        st.write(transformed_ground)
        st.write("Elevated Points:")
        st.write(elevated_points_np)

        # Optional: Visualize the alignment
        fig, ax = plt.subplots()
        ax.imshow(Image.open(elevated_image).convert("RGB"))
        ax.scatter(elevated_points_np[:,0], elevated_points_np[:,1], c='red', label='Elevated Points')
        ax.scatter(transformed_ground[:,0], transformed_ground[:,1], c='blue', marker='x', label='Transformed Ground Points')
        ax.legend()
        st.pyplot(fig)


