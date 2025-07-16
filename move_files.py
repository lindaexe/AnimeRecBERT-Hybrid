import shutil
import os

os.makedirs("Data/preprocessed/AnimeRatings_min_rating7-min_uc10-min_sc10-splitleave_one_out/", exist_ok=True)
source_path = 'Data/AnimeRatings/dataset.pkl'

destination_path = 'Data/preprocessed/AnimeRatings_min_rating7-min_uc10-min_sc10-splitleave_one_out/dataset.pkl'

shutil.move(source_path, destination_path)

source_path2 = 'Data/AnimeRatings/random-sample_size100-seed98765.pkl'

destination_path2 = 'Data/preprocessed/AnimeRatings_min_rating7-min_uc10-min_sc10-splitleave_one_out/random-sample_size100-seed98765.pkl'

shutil.move(source_path2, destination_path2)
