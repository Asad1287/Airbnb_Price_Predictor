import os
import pandas as pd

def run_etl():
    """
    Extract, transform, and load the Titanic dataset.
    """
    raw_data_path = os.path.join("data", "raw", "titanic.csv")
    processed_data_path = os.path.join("data", "processed", "titanic_processed.csv")

    # Extract
    print("Extracting data from:", raw_data_path)
    df = pd.read_csv(raw_data_path)

    # Transform
    print("Transforming data...")
    # Fill missing Age values with the median
    df['Age'].fillna(df['Age'].median(), inplace=True)
    # Map Sex to numeric values: male=1, female=0
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    # Fill missing Embarked values and map to numeric: C=0, Q=1, S=2
    df['Embarked'] = df['Embarked'].fillna('S')
    embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
    df['Embarked'] = df['Embarked'].map(embarked_mapping)

    # Load: Ensure the processed directory exists, then save the file.
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df.to_csv(processed_data_path, index=False)
    print("Processed data saved to:", processed_data_path)

if __name__ == "__main__":
    run_etl()
