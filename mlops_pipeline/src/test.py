import pandas as pd

# df = pd.read_excel("Base_de_datos.xlsx")
from google.cloud import bigquery
project_id = "pro-cientificos-acev"

client = bigquery.Client(project=project_id)
sql = """
SELECT * FROM `pro-cientificos-acev.financiero.scoring_creditos`
"""

df = client.query(sql).to_dataframe()  # devuelve pandas.DataFrame
print(df.shape)
print(df.columns.tolist())

corr = df.corr(numeric_only=True)["Pago_atiempo"].sort_values(ascending=False)
print("--------------------------------")
print("--------------------------------")
print(corr) 

print("--------------------------------")
print("Tests OK")
print("--------------------------------")
