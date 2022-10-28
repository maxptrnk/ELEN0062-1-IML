from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing


if __name__ == '__main__':

    # Fetching dataset
    dataset = fetch_california_housing()
    X, y = dataset.data, dataset.target

    # Scaling inputs
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Shuffling input-output pairs
    X, y = shuffle(X, y, random_state=42)

    # Your code here
    # ..............
