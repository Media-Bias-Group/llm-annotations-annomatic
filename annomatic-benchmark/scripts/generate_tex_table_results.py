# %%
import pandas as pd

d = pd.read_csv('benchmark_results.csv')

# %%
models = d.model_name.unique()[::-1]
run_types = ['0 Shot', '0 Shot + System Prompt', '0 Shot CoT', '2 Shot','4 Shot', '8 Shot', '2 Shot CoT', '4 Shot CoT', '8 Shot CoT']

# %%
metric = 'mcc'
print(run_types)
for model in models:
    df = d[d.model_name==model][['run_type',metric]]
    run_types_ = df['run_type'].tolist()

    print(model,end='')
    for run_type in run_types:
        if run_type not in run_types_:
            print(" & - ",end='')
        else:
            print(f" & {round(df[df['run_type']==run_type][metric].item(),3)}",end='')

    print(f" & {round(df[metric].mean(),3)}",end='')

    print("\\\\")


