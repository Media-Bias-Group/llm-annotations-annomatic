import pandas as pd

pool = pd.read_csv("reannotated_pool.csv")
pool = pool.rename(columns={'label': 'babe_label'})

pool['expert2'].fillna(pool['babe_label'].where(pool['babe_label'] == pool['gpt4_label']), inplace=True)
pool['expert1'].fillna(pool['babe_label'].where(pool['babe_label'] == pool['gpt4_label']), inplace=True)
pool['expert3'].fillna(pool['babe_label'].where(pool['babe_label'] == pool['gpt4_label']), inplace=True)
pool['expert4'].fillna(pool['babe_label'].where(pool['babe_label'] == pool['gpt4_label']), inplace=True)
pool['expert5'].fillna(pool['babe_label'].where(pool['babe_label'] == pool['gpt4_label']), inplace=True)

pool['label'] = pool[['babe_label', 'expert2', 'expert1', 'expert3', 'expert4', 'expert5']].mode(axis=1, dropna=True)[0]
pool['label_vergleich'] = pool[['babe_label', "gpt4_label", 'expert2', 'expert1', 'expert3',  'expert4', 'expert5']].mode(axis=1, dropna=True)[0]
pool['label'] = pool['label'].astype(int)

pool = pool.drop(columns=['babe_label', 'expert2', 'expert1', 'expert3', 'expert4', 'expert5', 'gpt4_label', 'label_vergleich'])
pool = pool.sample(frac=1, random_state=42).reset_index(drop=True)

pool['label'] = pool['label'].replace({0: 'NOT BIASED', 1: 'BIASED'})
pool.to_csv('final_pool.csv', index=False)


