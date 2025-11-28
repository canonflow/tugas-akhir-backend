import pandas as pd
import os

def create_sketch_dataframe(dataset_path, type, full_range=False):
    """
    Creates a DataFrame from the dataset folder structure for Siamese Neural Network training.
    Handles multiple images per sketch folder and multiple references per category.

    Args:
        dataset_path (str): Path to the main dataset directory (e.g., 'dataset')

    Returns:
        pd.DataFrame: DataFrame with columns ['Category', 'Reference Number', 'Reference Path', 'Sketch Path', 'Score']
    """
    data = []

    type_path = os.path.join(dataset_path, type)

    # Iterate through each category folder in train
    for category in os.listdir(type_path):
        category_path = os.path.join(type_path, category)

        if not os.path.isdir(category_path):
            continue

        print(f"--- Processing category: {category}")

        # Iterate through each reference folder in the category
        for ref_folder in os.listdir(category_path):
            ref_folder_path = os.path.join(category_path, ref_folder)

            if not os.path.isdir(ref_folder_path):
                continue

            # Extract reference number from folder name (e.g., "ref1" -> 1, "ref2" -> 2)
            ref_num = int(ref_folder[3:]) if ref_folder.startswith('ref') and ref_folder[3:].isdigit() else 0

            print(f"--- Processing reference: {ref_folder} (Ref #{ref_num})")

            # Define paths
            csv_path_string = 'scores-full.csv' if full_range else 'scores.csv'
            ref_image_path = os.path.join(ref_folder_path, 'ref.png')
            scores_csv_path = os.path.join(ref_folder_path, csv_path_string)
            sketches_path = os.path.join(ref_folder_path, 'sketches')

            # Check if required files exist
            if not os.path.exists(ref_image_path):
                print(f"     Missing ref.png: {ref_image_path}")
                continue
            if not os.path.exists(scores_csv_path):
                print(f"     Missing scores.csv: {scores_csv_path}")
                continue
            if not os.path.exists(sketches_path):
                print(f"     Missing sketches folder: {sketches_path}")
                continue

            # Load scores.csv (no header, first column = sketch folder name, second = score)
            try:
                scores_df = pd.read_csv(scores_csv_path, header=None, encoding='utf-8')
                # Clean up any whitespace in names
                scores_df[0] = scores_df[0].str.strip()
                scores_dict = dict(zip(scores_df[0], scores_df[1]))  # {person_name: score}
                print(f"      Loaded scores.csv with {len(scores_dict)} entries")
            except Exception as e:
                print(f"      Error reading scores.csv in {ref_folder_path}: {e}")
                continue

            # Iterate through each person's sketch folder
            for person_folder in os.listdir(sketches_path):
                person_path = os.path.join(sketches_path, person_folder)

                if not os.path.isdir(person_path):
                    continue

                # Get all image files in the person's folder
                sketch_files = []
                for f in os.listdir(person_path):
                    file_path = os.path.join(person_path, f)
                    if os.path.isfile(file_path):
                        sketch_files.append(f)

                if len(sketch_files) == 0:
                    print(f"      No sketch files found in {person_path}")
                    continue

                # Get score for this person from scores.csv
                score = scores_dict.get(person_folder)
                if score is None:
                    print(f"       No score found for {person_folder} in {scores_csv_path}")
                    continue

                # For each sketch file in this person's folder, create a row
                for sketch_file in sketch_files:
                    # Construct Sketch Path as requested: "dataset/sketches/<person>/<filename>"
                    sketch_path = f"{sketches_path}/{person_folder}/{sketch_file}"

                    curr_score = int(score) if full_range else float(score)

                    # Add row to data
                    data.append({
                        'Category': category,
                        'Reference Number': ref_num,
                        'Reference Path': ref_image_path,
                        'Sketch Path': sketch_path,
                        'Score': curr_score # Ensure score is numeric
                    })

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Category', 'Reference Number', 'Reference Path', 'Sketch Path', 'Score'])
    return df

def create_negative_pairs_from_positive_df(positive_df):
    """
    Creates balanced negative pairs DataFrame from the positive pairs DataFrame.
    Negative pairs are combinations of references and sketches from different categories.
    Score is always 0 for negative pairs.
    
    The number of negative pairs will match the number of positive pairs, distributed
    equally across all unique reference images.

    Args:
        positive_df (pd.DataFrame): DataFrame from create_sketch_dataframe() with positive pairs

    Returns:
        pd.DataFrame: DataFrame with columns ['Category', 'Reference Number', 'Reference Path', 'Sketch Path', 'Score']
    """
    import random
    
    # Get total number of positive pairs
    total_positive = len(positive_df)
    
    # Get unique references (each unique combination of Category + Reference Number + Reference Path)
    unique_refs = positive_df[['Category', 'Reference Number', 'Reference Path']].drop_duplicates().reset_index(drop=True)
    total_refs = len(unique_refs)
    
    # Get unique sketches for each category
    sketch_per_category = {}
    for category in positive_df['Category'].unique():
        cat_sketches = positive_df[positive_df['Category'] == category]['Sketch Path'].unique().tolist()
        sketch_per_category[category] = cat_sketches

    print(f"--- Total positive pairs: {total_positive}")
    print(f"--- Total unique references: {total_refs}")
    print(f"--- Found categories: {list(positive_df['Category'].unique())}")
    for cat in positive_df['Category'].unique():
        cat_refs = unique_refs[unique_refs['Category'] == cat]
        print(f"      {cat}: {len(cat_refs)} reference(s), {len(sketch_per_category[cat])} unique sketches")

    # Generate balanced negative pairs
    data = []
    categories = list(positive_df['Category'].unique())
    num_categories = len(categories)
    
    # Calculate how many negative pairs per reference
    negative_per_ref = total_positive // total_refs
    remainder = total_positive % total_refs  # Handle remainder
    
    # Calculate how many sketches to sample from each other category
    other_categories_count = num_categories - 1
    
    # Ensure we always sample at least 1 sketch per category if possible
    if other_categories_count > 0:
        sketches_per_other_category = max(1, negative_per_ref // other_categories_count)
    else:
        sketches_per_other_category = 0
    
    print(f"\n--- Balancing strategy:")
    print(f"     Target total negative pairs: {total_positive}")
    print(f"     Negative pairs per reference: {negative_per_ref} (remainder: {remainder})")
    print(f"     Other categories available: {other_categories_count}")
    print(f"     Sketches to sample per other category: {sketches_per_other_category}")

    # For each unique reference
    for idx, ref_row in unique_refs.iterrows():
        ref_category = ref_row['Category']
        ref_num = ref_row['Reference Number']
        ref_path = ref_row['Reference Path']
        
        # Get other categories (for negative pairing)
        other_categories = [cat for cat in categories if cat != ref_category]
        
        # Add extra pair for first 'remainder' references to reach exact total
        extra_pair = 1 if idx < remainder else 0
        total_for_this_ref = negative_per_ref + extra_pair
        
        if total_for_this_ref == 0:
            continue
            
        print(f"\n---- Processing ref: {ref_category} - ref{ref_num} (target: {total_for_this_ref} pairs)")
        
        # Distribute pairs across other categories
        pairs_collected = 0
        
        for i, sketch_category in enumerate(other_categories):
            available_sketches = sketch_per_category[sketch_category].copy()
            
            # Calculate how many to sample from this category
            remaining_pairs = total_for_this_ref - pairs_collected
            remaining_categories = len(other_categories) - i
            
            # Distribute remaining pairs evenly across remaining categories
            pairs_from_this_category = remaining_pairs // remaining_categories
            
            # Ensure at least 1 if we still need pairs
            if remaining_pairs > 0 and pairs_from_this_category == 0:
                pairs_from_this_category = 1
            
            if pairs_from_this_category > 0:
                # Randomly sample sketches (with replacement if needed)
                if len(available_sketches) >= pairs_from_this_category:
                    sampled_sketches = random.sample(available_sketches, pairs_from_this_category)
                else:
                    # If not enough sketches, sample with replacement
                    sampled_sketches = random.choices(available_sketches, k=pairs_from_this_category)
                
                print(f"      Sampling {len(sampled_sketches)} sketches from {sketch_category}")
                
                for sketch_path in sampled_sketches:
                    data.append({
                        'Category': ref_category,
                        'Reference Number': ref_num,
                        'Reference Path': ref_path,
                        'Sketch Path': sketch_path,
                        'Score': 0.0
                    })
                    pairs_collected += 1

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Category', 'Reference Number', 'Reference Path', 'Sketch Path', 'Score'])
    
    print(f"--- RESULT: Created {len(df)} negative pairs (target was {total_positive})")
    # print(f"{'='*50}")
    
    return df