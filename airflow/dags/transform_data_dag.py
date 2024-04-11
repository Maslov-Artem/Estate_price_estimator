from preprocessing import Preprocessor
from train_model import To_Model_BIG
import pandas as pd
from airflow.decorators import dag, task
from datetime import datetime
from airflow.providers.postgres.hooks.postgres import PostgresHook



default_args = {"owner": "Artem",
                "start_date": datetime(2024, 4, 4)}

@dag(dag_id="transform_data", default_args=default_args, schedule_interval="@daily",tags=['cian'])
def transform_data():

    @task
    def load_data_from_db():
        hook = PostgresHook(postgres_conn_id="estate_data")
        query = "SELECT * FROM real_estate"
        data = hook.get_pandas_df(sql=query)
        print(data.shape)
        data.to_csv("raw_data.csv")

    @task
    def preprocess_data():
        preprocessor = Preprocessor("raw_data.csv")
        preprocessor.transform()
        preprocessor.save("/opt/airflow/plugins/preprocessed_data.pkl")

        model = To_Model_BIG(path_or_DF=preprocessor.to_pandas(), k=1.7, test_size=0.2)
        model.fit()
        model.score()
        model.save("/opt/airflow/plugins/model.pkl")
    #     data = preprocessor.fit_transform(data)
    #     data = preprocessor_stage_2.fit_transform(data)
    #     data.to_csv("/opt/airflow/plugins/preprocessed_data.csv")
    #
    # @task
    # def train_model():
    #     data = pd.read_csv("/opt/airflow/plugins/preprocessed_data.csv")
    #     categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    #     model = CatBoostRegressor(cat_features=categorical_features)
    #     X = data.drop("price", axis=1)
    #     y = data["price"]
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
    #     mse = mean_absolute_error(y_test, y_pred)
    #     print("Mae: ", mse)
    #     model.save_model("/opt/airflow/plugins/catboostregressor.cbm", format="cbm")
    #
    # load_task = load_data_from_db()
    # preprocess_task = preprocess_data()
    # train_task = train_model()
    #
    # # Set up dependencies
    # preprocess_task << load_task
    # train_task << preprocess_task
    #
    # # load_data_from_db()
    # # preprocess_data()
    # # train_model()
    load_data_task = load_data_from_db()
    train_task = preprocess_data()
    load_data_task >> train_task

transform_data()

