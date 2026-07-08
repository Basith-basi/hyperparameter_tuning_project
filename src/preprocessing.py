import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, target_column):

    # -------------------------------
    # Handle Missing Values
    # -------------------------------

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())

    categorical_columns = df.select_dtypes(include=["object"]).columns

    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # -------------------------------
    # Encode Categorical Features
    # -------------------------------

    for col in categorical_columns:

        if df[col].nunique() == 2:

            encoder = LabelEncoder()

            df[col] = encoder.fit_transform(df[col])

        else:

            df = pd.get_dummies(
                df,
                columns=[col],
                drop_first=True
            )

    # -------------------------------
    # Split Features & Target
    # -------------------------------

    X = df.drop(columns=[target_column])

    y = df[target_column]

    # -------------------------------
    # Train-Test Split
    # -------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------------------------------
    # Optional Scaling
    # Uncomment if needed
    # -------------------------------

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test