import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

def main():
    df = pd.read_csv('data_sets/preprocessed_data.csv')

    y = df['Winner'].values

    stat_columns = df.select_dtypes(include=['number'])
    X = stat_columns.values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=50)) # 50 or 100 seem to work pretty well
    model.fit(X_train, y_train)
    print(model.score(X_valid, y_valid))

if __name__=='__main__':
    main()
