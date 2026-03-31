import numpy as np

def generate_data():

    # -------------------------------
    # 🔹 NORMAL USERS (MULTIPLE TYPES)
    # -------------------------------

    # High activity users
    normal1 = np.random.normal(
        loc=[5000, 30, 300, 10, 1, 0.5],
        scale=[1000, 5, 50, 3, 1, 0.2],
        size=(500, 6)
    )

    # Medium activity users
    normal2 = np.random.normal(
        loc=[3000, 25, 200, 7, 1, 0.3],
        scale=[800, 4, 40, 2, 1, 0.1],
        size=(300, 6)
    )

    # Low activity users
    normal3 = np.random.normal(
        loc=[1500, 20, 120, 4, 1, 0.2],
        scale=[500, 3, 30, 1, 1, 0.1],
        size=(200, 6)
    )

    normal = np.vstack([normal1, normal2, normal3])


    # -------------------------------
    # 🔹 FRAUD USERS (NORMAL FRAUD)
    # -------------------------------

    fraud1 = np.random.normal(
        loc=[50, 120, 10, 0, 5, 5],
        scale=[20, 10, 5, 1, 2, 1],
        size=(80, 6)
    )


    # -------------------------------
    # 🔹 EXTREME FRAUD USERS
    # -------------------------------

    fraud2 = np.random.normal(
        loc=[10, 200, 1, 0, 10, 10],
        scale=[5, 20, 2, 1, 3, 2],
        size=(40, 6)
    )

    fraud = np.vstack([fraud1, fraud2])


    # -------------------------------
    # 🔹 FINAL DATASET
    # -------------------------------

    data = np.vstack([normal, fraud])

    return data