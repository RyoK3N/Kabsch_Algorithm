# app.py

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

from utils import extract_line_points, calculate_angle
from kabsch import perform_kabsch

def main():
    st.title('Kabsch Algorithm')

    ground_image = st.file_uploader("Upload Ground View Image", type=["jpg", "jpeg", "png"], key="ground")
    elevated_image = st.file_uploader("Upload Elevated View Image", type=["jpg", "jpeg", "png"], key="elevated")

    if not ground_image or not elevated_image:
        st.info("Please upload both Ground and Elevated images to proceed.")
        return


    drawing_mode = st.sidebar.selectbox("Drawing tool:", ("point", "line",))

    ground_points = process_image(ground_image, "Ground", drawing_mode)
    elevated_points = process_image(elevated_image, "Elevated", drawing_mode)

    # Only when both i/ps are fed 
    if ground_points and elevated_points:
        st.subheader("Kabsch Algorithm Results")

        #Perform Kabsch
        results = perform_kabsch(ground_points, elevated_points)

        if results:
            R, t, transformed_ground, elevated_np, H = results

            st.write('Covariance Matrix')
            st.write(H)
            st.write("Rotation Matrix (R):")
            st.write(R)
            st.write("Translation Vector (t):")
            st.write(t)
            st.write("Transformed Ground Points:")
            st.write(transformed_ground)
            st.write("Elevated Points:")
            st.write(elevated_np)

            
            # Visualize the alignment
            fig, ax = plt.subplots()
            elevated_pil = Image.open(elevated_image).convert("RGB")
            ax.imshow(elevated_pil)
            ax.scatter(elevated_np[:,0], elevated_np[:,1], c='red', label='Elevated Points')
            ax.scatter(transformed_ground[:,0], transformed_ground[:,1], c='blue', marker='x', label='Transformed Ground Points')
            ax.legend()
            st.pyplot(fig)

def process_image(image, image_label, drawing_mode):
    if image is not None:
        image_pil = Image.open(image).convert("RGB")
        image_np = np.array(image_pil)

        img_width, img_height = image_pil.size

        st.subheader(f"{image_label} Image")

        # Sidebar settings
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
            fill_color="rgba(255, 165, 0, 0.3)",  
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=bg_image,
            update_streamlit=realtime_update,
            width=img_width,   
            height=img_height, 
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key=f"canvas_{image_label}",
        )

        points = []
        if canvas_result.json_data is not None:
            
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "circle" and drawing_mode == 'point':
                    cx, cy = obj["left"] + obj["radius"], obj["top"] + obj["radius"]
                    points.append((int(cx), int(cy)))

                # For freedraw
                # elif obj["type"] == "path" and drawing_mode == 'freedraw':
                #     path = obj.get("path", [])
                #     for point in path:
                #         if isinstance(point, list) and len(point) == 2:
                #             try:
                #                 x, y = float(point[0]), float(point[1])
                #                 points.append((int(x), int(y)))
                #             except ValueError:
                #                 # Skip non-numeric entries like 'M', 'L', etc.
                #                 continue

                # Handle 'line' type for 'line' mode
                elif obj["type"] == "line" and drawing_mode == 'line':
                    path = obj.get("path", [])
                    extracted_points = extract_line_points(path)
                    if len(extracted_points) == 2:
                        points.extend(extracted_points)
                    else:
                        st.warning(f"Invalid line detected on the {image_label} image. Please ensure you're drawing lines correctly.")

        # If in line mode, enforce that exactly two lines (four points) are drawn
        # if drawing_mode == 'line':
        #     if len(points) % 2 != 0:
        #         st.warning(f"Please draw complete lines on the {image_label} image.")
        #         return None

        if len(points) > 0:
            # Define point labels
            labels = [chr(65 + i) for i in range(len(points))]  # ASCII 

            image_np_copy = image_np.copy()

            # Draw points, labels, and lines
            for i, (point, label) in enumerate(zip(points, labels)):
                cv2.putText(image_np_copy, label, (point[0] + 10, point[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Draw lines connecting the points based on drawing mode
            if drawing_mode == 'line':
                for i in range(0, len(points), 2):
                    if i + 1 < len(points):
                        cv2.line(image_np_copy, points[i], points[i+1], (255, 0, 0), 2)
            else:
                # ### other mode ##remove
                for i in range(1, len(points)):
                    cv2.line(image_np_copy, points[i - 1], points[i], (255, 0, 0), 1)

            # Ensure at least 3 points are present to calculate angles
            if len(points) >= 3:
                # Calculate and draw angles between consecutive lines
                for i in range(1, len(points) - 1):
                    angle = calculate_angle(points[i - 1], points[i], points[i + 1])
                    angle_text = f"{angle:.1f}°"
                    
                    # Display the angle at B
                    cv2.putText(image_np_copy, angle_text, (points[i][0] - 30, points[i][1] + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Display the output image
            st.subheader(f"{image_label} Image with Annotations")
            st.image(image_np_copy, use_column_width=True)

            # Display coordinates of points with labels
            labeled_points = {label: point for label, point in zip(labels, points)}
            st.write(f"Coordinates of Points with Labels for {image_label} Image:", labeled_points)
        else:
            st.write(f"No points drawn yet on the {image_label} image.")

        return points  

    return None

if __name__ == "__main__":
    main()
