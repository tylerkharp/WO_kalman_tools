import annotation_to_metric as atm
import pickle
import pandas as pd

def load_annotations_as_df(pickle_file):
    path = pickle_file
    with open(path, "rb") as fobj:
        df = pd.DataFrame(pickle.load(fobj))
    return df
def load_annotations_as_dict(pickle_file):
    path = pickle_file
    with open(path, "rb") as fobj:
        dict = pickle.load(fobj)
    return dict

def remove_annotations_by_string(df,remove_string):
    for run_name in df.iloc[:,0]:
        if remove_string in run_name:
            df = df[df.iloc[:,0] != run_name]
    return df

def save_var_as_pickle(df,file_name):    
    with open(file_name + '.pkl', 'wb') as file:
        pickle.dump(df, file)

def combine_pickle_files(file_name,file_list):
    df_list = []
    for file in file_list:
        df_list.append(load_annotations_as_df(file))
    df = pd.concat(df_list)
    destination = file_list[0][0:file_list[0].rfind('/')]
    with open(destination + '/' + file_name + '.pkl', 'wb') as file:
        pickle.dump(df, file)
    return df

if __name__ == "__main__":
    df = atm.load_annotations_as_df("indications.pkl")
    df = remove_annotations_by_string(df,'G4-0047')
    save_var_as_pickle(df,'indications')