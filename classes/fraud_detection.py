import pandas            as pd
import numpy             as np
import pickle
import json
from category_encoders import OneHotEncoder


class Fraud_Detection(object):

    def __init__(self):

        self.amount_scaler = pickle.load(open('parameters/amount_scaler.pkl', 'rb'))
        self.flow_dest_scaler = pickle.load(open('parameters/flow_dest_scaler.pkl', 'rb'))
        self.flow_orig_scaler = pickle.load(open('parameters/flow_orig_scaler.pkl', 'rb'))
        self.new_balance_dest_scaler = pickle.load(open('parameters/new_balance_dest_scaler.pkl', 'rb'))
        self.new_balance_orig_scaler = pickle.load(open('parameters/new_balance_orig_scaler.pkl', 'rb'))
        self.step_scaler = pickle.load(open('parameters/step_scaler.pkl', 'rb'))
        self.ohe = pickle.load(open('parameters/onehotencoder_amount_level.pkl', 'rb'))
        self.manual_selected_features = pickle.load(open('parameters/manual_selected_features.pkl', 'rb'))

    def features_engineering(self, df):

        '''creating and transforming dataset's features'''

        # creating the dataset's new columns names
        columns = ['step', 'type', 'amount', 'nameOrig', 'OldBalanceOrig', 'NewBalanceOrig',
                   'nameDest', 'oldBalanceDest', 'NewBalanceDest', 'IsFraud', 'IsFlaggedFraud']

        # changing the dataset's columns names for the 'new_columns' list values
        snakecase = lambda x: inflection.underscore(x)
        new_columns = list(map(snakecase, columns))
        df1.columns = new_columns

        # creating a dataset's column with the day
        df['day'] = df['step'].apply(lambda x: int(x / 24 + 1))

        # creating a dataset's column with the day of the week type (weekend or workweek)
        df['weekend'] = df['day'].apply(lambda x:
                                        1 if x in [6, 7, 13, 14, 20, 21, 27, 28] else
                                        0)

        # creating a dataset's column with difference (amount) between the old and new origin account balance
        df['flow_orig'] = df['new_balance_orig'] - df['old_balance_orig']

        # creating a dataset's column with difference (amount) between the old and new destination account balance
        df['flow_dest'] = df['new_balance_dest'] - df['old_balance_dest']

        # resetting the dataset's original index to 0
        df.reset_index(drop=True, inplace=True)

        return df

    def data_preparation(self, df):

        ''' preparing the dataset's features '''

        # rescaling numerical features
        df['amount'] = self.amount_scaler.transform(df[['amount']].values)
        df['flow_dest'] = self.flow_dest_scaler.transform(df[['flow_dest']].values)
        df['flow_orig'] = self.flow_orig_scaler.transform(df[['flow_orig']].values)
        df['new_balance_dest'] = self.new_balance_dest_scaler.transform(df[['new_balance_dest']].values)
        df['new_balance_orig'] = self.new_balance_orig_scaler.transform(df[['new_balance_orig']].values)
        df['step'] = self.step_scaler.transform(df[['step']].values)

        # deleting the is_fraud column, because it is the answer and it will bug the OneHotEncoder method it it remains in the dataset
        df.drop(columns=['is_fraud'], inplace=True)

        # transforming the 'df' dataset with the 'ohe' method
        df = self.ohe.transform(df)

        # renaming the columns created by the OneHotEncoder method
        df.rename(columns={'type_CASH_OUT': 'cash_out',
                           'type_PAYMENT': 'payment',
                           'type_CASH_IN': 'cash_in',
                           'type_TRANSFER': 'transfer',
                           'type_DEBIT': 'debit'},
                  inplace=True)

        return df

    def get_predict(self, model, df):

        ''' Filtering the features that will be used to predict and predict '''

        # selecting only the features that will be used to predict the result
        selected_features = self.manual_selected_features

        df_copy = df[selected_features].copy()

        # getting prediction
        df_copy['predict'] = model.predict(df_copy)

        # manipulating the return
        if df_copy['predict'][0] == 1:
            df_copy['answer'] = 'This is a fraudulent transaction'
        if df_copy['predict'][0] == 0:
            df_copy['answer'] = 'This is not a fraudulent transaction'

        return df_copy['answer'].to_json(orient='records')