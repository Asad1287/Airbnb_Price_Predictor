import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    """
    Train a logistic regression model on the processed Titanic dataset.
    """
    processed_data_path = os.path.join("data", "processed", "titanic_processed.csv")
    model_output_path = os.path.join("models", "titanic_model.pkl")
    
    # Load processed data
    print("Loading processed data from:", processed_data_path)
    df = pd.read_csv(processed_data_path)
    
    # Define features and target
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target = 'Survived'
    X = df[features]
    y = df[target]
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the logistic regression model
    print("Training model...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model accuracy:", accuracy)
    
    # Save the model (ensure the models directory exists)
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, "wb") as f:
        pickle.dump(model, f)
    print("Model saved to:", model_output_path)
    
if __name__ == "__main__":
    train_model()
