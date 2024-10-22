import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

RAW_DATA_PATH = "Data\iris\iris.data"
DEFAULT_STORAGE_DATA_PATH = "Data\processed"

def precess_raw_dataset():
    """
    数据预处理，将数据打乱顺序，并存储到默认路径。
    """
    lines = []
    with open(RAW_DATA_PATH, "r") as f:
        for line in f:
            if (line.strip() == ""):
                continue
            lines.append(line)
    
    import random
    random.shuffle(lines)

    with open(DEFAULT_STORAGE_DATA_PATH + "\iris_precessed.data", "w") as f:
        for line in lines:
            f.write(line)

if __name__ == "__main__":
    precess_raw_dataset()