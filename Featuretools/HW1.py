'''
    file name: PHW1
    author: Ji Woo Kim
    modified: 2022.04.05
'''
import pandas as pd
import numpy as np
# feature tools for automated feature engineering
import featuretools as ft

# Reading the data
clients_df = pd.read_csv('data/clients.csv', parse_dates = ['joined'])
loans_df = pd.read_csv('data/loans.csv', parse_dates = ['loan_start', 'loan_end'])
payments_df = pd.read_csv('data/payments.csv', parse_dates = ['payment_date'])

# Create new Entity set
es = ft.EntitySet(id='clients')
es = es.add_dataframe(
    dataframe_name='clients',
    dataframe=clients_df,
    index='client_id',
    time_index='joined')

es = es.add_dataframe(
    dataframe_name='payments',
    dataframe=payments_df,
    make_index=True,
    index='payment_id',
    time_index='payment_date')

es = es.add_dataframe(
    dataframe_name='loans',
    dataframe=loans_df,
    index='loan_id',
    time_index='loan_start')

es = es.add_dataframe(
    dataframe_name='total_loan_amount',
    dataframe=loans_df,
    make_index=True,
    index='total_loan_amount',
    time_index='loan_start')

'''
print(es['clients'])
print(es['payments'])
print(es['loans'])
'''

stats = loans_df.groupby('client_id')['loan_amount'].agg(['mean','max','min'])
stats.columns = ['mean_loan_amount','max_loan_amount','min_loan_amount']

# merge with the clients dataframe
stats = clients_df.merge(stats,left_on='client_id',right_index=True, how = 'left')
stats.head(10)

print(stats)

# r_client_previous = ft.Relationship(entityset=es, parent_dataframe_name="clients", parent_column_name="client_id", child_dataframe_name ="loans", child_column_name="client_id")
# es = es.add_relationship(r_client_previous)

r_payments = ft.Relationship(es,'loans','loan_id','payments','loan_id')
es = es.add_relationship(r_payments)

print(es['payments'])







