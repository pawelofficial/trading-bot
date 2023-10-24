import my_pgsql
from torch_model4 import Network
import torch
import pandas as pd 
from torch.utils.data import Dataset, DataLoader, random_split
    # reads row from pgsql, calls NN to get output 
def test__pg_to_model_output(
    model_fp='./models/wave_models/wave_loop.pth'
    ,N=5
):

    p=my_pgsql.mydb()
    df=p.read_df_array(tbl='quantiles',last_n_rows=N)
    df.to_csv('./data/foo.csv',sep='|')
    df=pd.read_csv('./data/quantiles_df.csv',sep='|')
    print(df.shape)
    exit(1)
    # cast all columns with ar_  to a  list 
    ar_columns=[x for x in df.columns.tolist() if 'ar_' in x]
    l=df[ar_columns].values.tolist()[0]
    model=Network(len(l),1,scale_factor=2)
    model.load_state_dict(torch.load(model_fp))

    X = torch.tensor( l ).float().view(1, -1)
    
    val_loader = DataLoader(X, batch_size=32, shuffle=False)  # Make sure shuffle is False
    outputs_list = []
    
    for data in val_loader:
        outputs = model(data)
        outputs_squeezed = outputs.squeeze().tolist()
        if isinstance(outputs_squeezed, float):
            outputs_list.append(outputs_squeezed)
        else:
            outputs_list.extend(outputs_squeezed)

        print(outputs_list)

    pass 

def test__pgsql_ar_insert():
    p=my_pgsql.mydb()
    p.execute_dml(query='truncate table quantiles')
    df=pd.read_csv('./data/quantiles_df.csv',sep='|').iloc[:100]
    p.write_df_array(df=df,tgt_ar_col='ar',tbl='quantiles')
    df=p.execute_select(query='select  array_length(ar,1) from quantiles limit 1')
    print(df)
    
    
    
    

if __name__=='__main__':
#    df=pd.read_csv('./data/quantiles_df.csv',sep='|')
#    print(df.shape)
  #  test__pgsql_ar_insert()
    test__pg_to_model_output()