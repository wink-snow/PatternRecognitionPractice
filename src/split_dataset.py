
def split_dataset(dataset: list, split_ratio: float) -> list:
    """
    针对交叉验证，将数据集划分为训练集和测试集。

    Parameters:
        `dataset` (list): 待划分的数据集
        `split_ratio` (float): 划分比例

    Returns:
        `splited_dataset` (list): 划分后的数据集

    """
    step = int(len(dataset) * split_ratio)
    splited_dataset = [dataset[i: i + step] for i in range(0, len(dataset), step)]
    return splited_dataset

if __name__ == '__main__':
    dataset = [[1, 2], [3, 4], [5, 6], [7, 8]]
    print(split_dataset(dataset, 0.8))