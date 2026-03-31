import numpy as np

def generate_data():
    # Normal users
    normal = np.random.normal(
        loc=[5000, 30, 300, 10, 1, 0.5], 
        scale=[1000, 5, 50, 3, 1, 0.2], 
        size=(200,6)
    )

    # Fraud users
    fraud = np.random.normal(
        loc=[50, 120, 10, 0, 5, 5], 
        scale=[20, 10, 5, 1, 2, 1], 
        size=(20,6)
    )

    data = np.vstack([normal, fraud])
    return data