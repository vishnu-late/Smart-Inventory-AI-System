import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data():
    np.random.seed(42)
    products = {
        'P001': 'Laptop',
        'P002': 'Smartphone',
        'P003': 'Headphones',
        'P004': 'Smartwatch',
        'P005': 'Tablet'
    }
    
    data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for _ in range(1000):
        date = start_date + timedelta(days=np.random.randint(0, 365))
        prod_id = np.random.choice(list(products.keys()))
        quantity = np.random.randint(1, 20)
        stock = np.random.randint(0, 100)
        
        data.append([date.strftime('%Y-%m-%d'), prod_id, quantity, stock])
        
    df = pd.DataFrame(data, columns=['Date', 'Product_ID', 'Quantity', 'Stock'])
    df.to_csv('data/sales_data.csv', index=False)
    print("Sample data generated at data/sales_data.csv")

if __name__ == "__main__":
    generate_data()
