import pandas as pd

pool = pd.read_csv("reannotated_pool.csv")
pool = pool.rename(columns={'label': 'babe_label'})

pool['Christoph'].fillna(pool['babe_label'].where(pool['babe_label'] == pool['gpt4_label']), inplace=True)
pool['Tomas'].fillna(pool['babe_label'].where(pool['babe_label'] == pool['gpt4_label']), inplace=True)
pool['Timo'].fillna(pool['babe_label'].where(pool['babe_label'] == pool['gpt4_label']), inplace=True)
pool['Martin'].fillna(pool['babe_label'].where(pool['babe_label'] == pool['gpt4_label']), inplace=True)
pool['Smi'].fillna(pool['babe_label'].where(pool['babe_label'] == pool['gpt4_label']), inplace=True)

pool['label'] = pool[['babe_label', 'Christoph', 'Tomas', 'Timo', 'Martin', 'Smi']].mode(axis=1, dropna=True)[0]
pool['label_vergleich'] = pool[['babe_label', "gpt4_label", 'Christoph', 'Tomas', 'Timo',  'Martin', 'Smi']].mode(axis=1, dropna=True)[0]
pool['label'] = pool['label'].astype(int)

pool = pool.drop(columns=['babe_label', 'Christoph', 'Tomas', 'Timo', 'Martin', 'Smi', 'gpt4_label', 'label_vergleich'])
pool = pool.sample(frac=1, random_state=42).reset_index(drop=True)

pool['label'] = pool['label'].replace({0: 'NOT BIASED', 1: 'BIASED'})
pool.to_csv('final_pool.csv', index=False)


