import pandas as pd


def csv_df(path):
    df = pd.read_csv(path)
    return df


def preprocess(df):
    return


def main():
    expr_df = csv_df('data/CCLE_expression.csv')    # 1406 rows x 19222 columns]
    info_df = csv_df("data/sample_info.csv")
    print(expr_df)
    print(info_df)


if __name__ == "__main__":
    main()
