import Augmentor
import os
import shutil

def augment_data(path):
    p = Augmentor.Pipeline(path)
    p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
    p.zoom(probability=0.65, min_factor=1.05, max_factor=1.25)
    p.random_distortion(probability=0.8, grid_width=30, grid_height=30, magnitude=14)
    p.skew(probability=0.6)
    p.shear(probability=0.7, max_shear_left=2, max_shear_right=2)

    p.sample(9)

def augment_pipeline(dirs, new_reference_name):
    for dir in dirs:
        curr_path = f"data/dataset/train/{new_reference_name}/ref1/sketches/{dir}"
        print("--- 01. Augment the images")
        augment_data(curr_path)

        # TODO: Since the output will be in the 'output' directory, we need to move the augmented images out
        print("--- 02. Move the output")
        output_path = f"{curr_path}/output"
        for filename in os.listdir(output_path):
            src_path = os.path.join(output_path, filename)

            if os.path.isfile(src_path):
                shutil.move(src_path, curr_path)
            print(f"---- Move: {filename}")

        print("--- 03. Remove the output directory")
        os.rmdir(output_path)