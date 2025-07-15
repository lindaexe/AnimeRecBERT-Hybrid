from .ml_20m import ML20MDataset
from .AnimeRatings54m import AnimeRatingsDataset

DATASETS = {
    ML20MDataset.code(): ML20MDataset,
    "AnimeRatings54M": AnimeRatingsDataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
