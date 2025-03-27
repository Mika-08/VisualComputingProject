import os
import pickle
import Test_single
import time

# Paths
test_folder = "Enhanced_Images/Blurred_Images/Vertical_2/Blur_90"
pickle_file = "joints_data_2.pkl"  # The file where extracted joints are stored

# Load pre-saved joint data from the pickle file
with open(pickle_file, "rb") as f:
    joints_dict = pickle.load(f)

total_images = 0
total_mse = 0
start_time = time.time()
current_amount = 0

for file in os.scandir(test_folder):
    current_amount += 1
    if current_amount % 10 == 0:
        print(current_amount)
    if file.is_file():
        file_path = file.path
        if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image_name = os.path.basename(file_path)
            print(image_name)

            # Get ground truth joints from the loaded dictionary
            ground_truth = joints_dict.get(image_name, None)

            if ground_truth is None:
                print(f"No ground truth found for {image_name}, skipping...")
                continue

            try:
                predictions = Test_single.run_model(file_path)
                mse = Test_single.calculate_mse(predictions, ground_truth)
                print(f"MSE for {image_name}: {mse}")
            except AttributeError:
                print(f"Error processing {image_name}, skipping...")
                continue

            if mse is not None:
                total_mse += mse
                total_images += 1

end_time = time.time()

# Final results
output_folder = "Output"
output_path = os.path.join(output_folder, test_folder)


print("Total time:", end_time - start_time)
print("Total images:", total_images)
print("Total mse:", total_mse)
print("Average MSE:", total_mse / total_images if total_images > 0 else "N/A")

with open(output_path, "w") as f:
    f.write(f"Total time: {end_time - start_time}")
    f.write(f"Total images: {total_images}")
    f.write(f"Total mse: {total_mse}")
    f.write(f"Average mse: {total_mse/total_images}")
