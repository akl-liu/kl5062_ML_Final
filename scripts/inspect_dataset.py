from datasets import load_dataset
import pandas as pd


def check_available_splits(dataset_name, required_splits=("train", "validation")):
    """
    Load the dataset without a split filter to discover which splits actually
    exist, and fail with a clear message instead of a raw traceback if a
    required split is missing.
    """
    dataset_dict = load_dataset(dataset_name)
    available_splits = list(dataset_dict.keys())
    print(f"\n Available splits for '{dataset_name}': {available_splits}")

    missing_splits = [s for s in required_splits if s not in available_splits]
    if missing_splits:
        raise SystemExit(
            f"The dataset '{dataset_name}' does not contain split(s): {missing_splits}. "
            f"Available splits are: {available_splits}"
        )

    return available_splits


def main():
    dataset_name = "RIW/small-coco"

    check_available_splits(dataset_name)

    #load the dataset
    train_ds = load_dataset(dataset_name,split="train")
    val_ds = load_dataset(dataset_name,split="validation")

    print("\n Dataset loaded successfully!")

    print("\n Train Dataset:")
    print(train_ds)

    print("\n Validation Dataset:")
    print(val_ds)

    print("\n Columns:")
    print(train_ds.column_names)


    print("\nFirst 5 samples from the train set:")
    print(train_ds[:5]) 

    for key, value in train_ds.features.items():
        print(f"\nFeature name: {key}")
        print(f"Feature type: {value}")


if __name__ == "__main__":
    main()  



