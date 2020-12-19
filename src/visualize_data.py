from os.path import dirname, abspath, join
import pandas as pd


if __name__ == "__main__":
    data_path = join(dirname(dirname(abspath(__file__))), "data", "TextComplexityDE19")
    ratings_df = pd.read_csv(join(data_path, "ratings.csv"), sep = ",", encoding = "ISO-8859-1")
    print(ratings_df.head())
