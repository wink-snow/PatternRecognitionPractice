import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def fetch_dataset(file_path: str):
    """
    Fetches the dataset from the given file path.
    Returns the features and labels as separate lists.
    Parameters:
        `file_path`: Path to the dataset file.
        Default value `DATA_PATH` is "Data\iris\iris.data".
    Returns:
        `data_X`: List of features.
        `data_y`: List of labels.
    """
    data_X = []
    data_y = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            processed_line_list = (line.strip()).split(',')
            data_X.append(list(float(processed_line_list[i]) for i in range(len(processed_line_list) - 1)))
            data_y.append(processed_line_list[-1])
    
    return data_X, data_y

if __name__ == "__main__":
    X, y = fetch_dataset(file_path = "Data\processed\iris_precessed.data")

    print(X)
    print(y)