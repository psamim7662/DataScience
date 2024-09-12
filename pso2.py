import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# PSO Parameters
SWARM_SIZE = 30    # Number of particles
DIMENSIONS = 3     # Number of thresholds (for example, 3 thresholds for 4 classes)
MAX_ITERATIONS = 1  # Maximum number of iterations
W = 0.5    # Inertia weight
C1 = 1.5   # Cognitive coefficient
C2 = 1.5   # Social coefficient
RANGE = 256  # Range for initializing particle positions (pixel intensity range)

# Objective function to minimize (Otsu's between-class variance)
def otsu_objective_function(thresholds, image_histogram, total_pixels):
    thresholds = np.sort(thresholds).astype(int)  # Sort and convert thresholds to integers
    thresholds = np.concatenate(([0], thresholds, [255]))  # Add 0 and 255 as boundaries
    between_class_variance = 0
    global_mean = np.sum(np.arange(256) * image_histogram) / total_pixels  # Overall mean of the image
    
    for i in range(1, len(thresholds)):
        class_range = image_histogram[thresholds[i-1]:thresholds[i]]
        class_pixels = np.sum(class_range)
        
        if class_pixels == 0:
            continue
        
        p_class = class_pixels / total_pixels  # Class probability
        mean_class = np.sum(np.arange(thresholds[i-1], thresholds[i]) * class_range) / class_pixels
        between_class_variance += p_class * (mean_class - global_mean) ** 2
    
    return -between_class_variance if between_class_variance > 0 else np.inf

# PSO algorithm for thresholding
def pso_algorithm(image_histogram, total_pixels):
    swarm = np.random.uniform(0, RANGE, (SWARM_SIZE, DIMENSIONS))  # Initialize particle positions
    velocities = np.random.uniform(-RANGE, RANGE, (SWARM_SIZE, DIMENSIONS))  # Initialize particle velocities
    personal_best_positions = np.copy(swarm)
    personal_best_values = np.apply_along_axis(otsu_objective_function, 1, swarm, image_histogram, total_pixels)
    
    global_best_index = np.argmin(personal_best_values)
    global_best_position = personal_best_positions[global_best_index]
    global_best_value = personal_best_values[global_best_index]
    
    for t in range(MAX_ITERATIONS):
        # Update velocities and positions
        r1, r2 = np.random.rand(SWARM_SIZE, DIMENSIONS), np.random.rand(SWARM_SIZE, DIMENSIONS)
        velocities = (W * velocities +
                      C1 * r1 * (personal_best_positions - swarm) +
                      C2 * r2 * (global_best_position - swarm))
        swarm += velocities
        swarm = np.clip(swarm, 0, RANGE)  # Ensure thresholds are within the valid range
        
        fitness_values = np.apply_along_axis(otsu_objective_function, 1, swarm, image_histogram, total_pixels)
        
        # Update personal bests
        for i in range(SWARM_SIZE):
            if fitness_values[i] < personal_best_values[i]:
                personal_best_positions[i] = swarm[i]
                personal_best_values[i] = fitness_values[i]

        # Update global best
        current_best_index = np.argmin(personal_best_values)
        current_best_value = personal_best_values[current_best_index]
        if current_best_value < global_best_value:
            global_best_position = personal_best_positions[current_best_index]
            global_best_value = current_best_value

    return global_best_position, global_best_value

# Apply the optimal thresholds to the image
def apply_thresholds(image, thresholds):
    thresholds = np.sort(thresholds).astype(int)
    segmented_image = np.zeros_like(image)
    
    # Apply thresholds to create segmented image
    segmented_image[image <= thresholds[0]] = 0
    for i in range(1, len(thresholds)):
        segmented_image[(image > thresholds[i-1]) & (image <= thresholds[i])] = i
    segmented_image[image > thresholds[-1]] = len(thresholds)
    
    return segmented_image

# Process all images in a folder
def process_images_from_folder(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path, 0)  # Load image in grayscale
            
            if image is None:
                print(f"Could not read {filename}. Skipping.")
                continue
            
            # Compute image histogram
            image_histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
            total_pixels = image.size
            
            # Run PSO for multi-level Otsu thresholding
            best_position, best_value = pso_algorithm(image_histogram, total_pixels)
            print(f"Processing {filename}")
            print(f"Best Thresholds: {np.sort(best_position)}")
            print(f"Best Value: {best_value}")
            
            # Apply thresholds to the image
            segmented_image = apply_thresholds(image, best_position)
            
            # Plot the original and segmented images side by side
            plt.figure(figsize=(10, 5))
            
            # Original image
            plt.subplot(1, 2, 1)
            plt.title(f'Original Image: {filename}')
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            
            # Segmented image
            plt.subplot(1, 2, 2)
            plt.title(f'Segmented Image: {filename}')
            plt.imshow(segmented_image, cmap='gray')
            plt.axis('off')
            
            # Display the plot
            plt.show()

if __name__ == "__main__":
    input_folder = "dataset"  # Folder containing the images
    process_images_from_folder(input_folder)
