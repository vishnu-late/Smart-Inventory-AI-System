import pandas as pd
import pickle
from utils.preprocessing import load_data, clean_data, add_features
from utils.model_utils import train_model

def main():
    # Load and clean data
    df = load_data('data/sales_data.csv')
    if df is not None:
        df = clean_data(df)
        df = add_features(df)
        
        # Train model
        train_model(df)
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()
