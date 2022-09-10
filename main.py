import pandas as pd
import torch
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Union, Iterable
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from mlp import MLP
from tqdm.auto import tqdm


class CustomizedDataset(Dataset):
    def __init__(self, data_df, label_df, per):
        data_temp = torch.tensor(data_df.values, dtype=torch.float32)
        z_score = (data_temp - data_temp.mean()) / data_temp.std()
        min_max_normalization = (z_score - z_score.min()) / (z_score.max() - z_score.min())
        if per > 0:
            self.x = min_max_normalization[:int(min_max_normalization.size(0) * per)]
        else:
            self.x = min_max_normalization[int(min_max_normalization.size(0) * (1 + per)):]

        temp = np.array(label_df.values)
        temp = temp.flatten()
        tokens = [get_tokenizer('basic_english')(str(w)) for w in temp]
        vocabulary = build_vocab_from_iterator(tokens)

        def one_hot_encoding_from_text(vset, keys: Union[str, Iterable]):
            if isinstance(keys, str):
                keys = [keys]
            return func.one_hot(torch.tensor(vset(keys)), num_classes=len(vset))

        def tissue_type_one_hot_encoding(vset, flattened_list):
            list_encoded = []
            for i in range(0, len(flattened_list)):
                encoded = one_hot_encoding_from_text(vset, str(flattened_list[i]).lower())
                encoded = torch.reshape(encoded, (-1, ))
                list_encoded.append(encoded.tolist())
            return torch.tensor(list_encoded, dtype=torch.float32)

        label = tissue_type_one_hot_encoding(vocabulary, temp)
        if per > 0:
            self.y = label[:int(label.size(0) * per)]
        else:
            self.y = label[int(label.size(0) * (1 + per)):]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def csv_df(path):
    df = pd.read_csv(path)
    return df


def preprocess(path_d, path_l):
    expr_df_raw = csv_df(path_d)    # 1406 rows x 19222 columns
    info_df_raw = csv_df(path_l)    # 1840 rows x 29 columns
    info_df = pd.DataFrame(info_df_raw[['DepMap_ID', 'sample_collection_site']])
    dict_depmap_id = {}
    for i, r in expr_df_raw.iterrows():
        dict_depmap_id[str(r.iloc[0])] = i
    for i, r in info_df.iterrows():
        if not (str(r.iloc[0]) in dict_depmap_id):
            info_df = info_df.drop([i])
    # info_df 1406 rows x 2 columns

    reordered_list = []
    for i, r in expr_df_raw.iterrows():
        temp = info_df[info_df['DepMap_ID'] == str(r.iloc[0])].iloc[-1, -1]
        reordered_list.append(temp)
    expr_df = expr_df_raw.iloc[:, 1:]   # 1406 rows x 19221 columns
    tissue_df = pd.DataFrame(reordered_list, columns=['sample_collection_site'])    # 1406 rows x 1 columns
    return expr_df, tissue_df


def run(model, loss_func, optimizer, dataloader, batch_size, epoch, is_train):
    t_loader = tqdm(enumerate(dataloader), total=len(dataloader))
    loss_final = 0
    for i, (x, y) in t_loader:
        x = x.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        # loss = torch.nn.functional.binary_cross_entropy(y_pred, y)

        if is_train:
            loss.backward()
            optimizer.step()

        t_loader.set_description(
            'Epoch {} {} Loss: {:.4f}'.format(epoch, (
                'Train' if is_train else 'Test '), loss.item()))

        loss_final += loss.item()
    loss_final = loss_final / len(dataloader)

    if not is_train:
        print('TESTING SET RESULTS: Average loss: {:.4f}'.format(
            loss_final))

    torch.cuda.empty_cache()


@torch.no_grad()
def compute_accuracy():
    return


def train(model, loss, optimizer, train_loader, batch_size, epoch):
    run(model, loss, optimizer, train_loader, batch_size, epoch, is_train=True)


@torch.no_grad()
def test(model, loss, optimizer, test_loader, batch_size, epoch):
    run(model, loss, optimizer, test_loader, batch_size, epoch, is_train=False)


def main():
    batch_size = 10
    lr = 0.002
    epoch_num = 40

    data_df, label_df = preprocess(path_d='data/CCLE_expression.csv', path_l='data/sample_info.csv')
    train_dataset = CustomizedDataset(data_df, label_df, 0.9)
    test_dataset = CustomizedDataset(data_df, label_df, -0.1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    m = train_dataset[0][0].shape[0]
    n = train_dataset[0][1].shape[0]
    model = MLP(m, n)
    model = model.cuda()
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    test(model, loss, optimizer, train_loader, batch_size, 0)

    for epoch in range(1, epoch_num + 1):
        train(model, loss, optimizer, train_loader, batch_size, epoch)
        test(model, loss, optimizer, test_loader, batch_size, epoch)


if __name__ == "__main__":
    main()
