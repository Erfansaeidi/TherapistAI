import pandas as pd
from datasets import Dataset

def load_and_preprocess_data(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Perform any necessary preprocessing steps
    # For example, renaming columns, handling missing values, etc.
    # df = df.rename(columns={'old_name': 'new_name'})

    # Convert the DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    return dataset

if __name__ == "__main__":
    dataset = load_and_preprocess_data("../dataset/test.csv")
    print(dataset)