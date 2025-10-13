import h5py
import os

h5_file_name = '0.h5'

if os.path.exists(h5_file_name):
    try:
        with h5py.File(h5_file_name, 'r') as f:
            print(f"---- FileName: {os.path.basename(h5_file_name)} ----")
            print("Key List:", list(f.keys()))

            if (list(f.keys())):
                print(f"Key name: {list(f.keys())[0]}")
                print(f"Data Shape: {f[list(f.keys())[0]].shape}")
                print(f"The number of Dim: {f[list(f.keys())[0]].ndim}")
            
                if f[list(f.keys())[0]].ndim == 1:
                    print("-> This key is most likely an embedding vector.")
                elif f[list(f.keys())[0]].ndim == 2 and f[list(f.keys())[0]].shape[0] == 1:
                    print("-> This key is most likely an embedding vector with shape (1, D).")
                else:
                    print("-> This key may has multiple embedding, or be other type data.")
    except Exception as e:
        print(f"Exception occur.")
else:
    print("The file does not exists.")