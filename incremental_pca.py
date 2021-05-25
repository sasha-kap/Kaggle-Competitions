"""
1. Read S3 CSV files
2. Iterate over CSV chunks and apply StandardScaler (two passes: fit and transform)
(per https://stackoverflow.com/questions/52642940/feature-scaling-for-a-big-dataset)
3. Pass scaled data to IncrementalPCA() (partial_fit, then transform on the third pass
over CSV chunks)
"""

from itertools import tee

import boto3
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

bucket = "my-rds-exports"
file_name = "csv1.csv"

s3_client = boto3.client("s3")
response = s3_client.get_object(Bucket=bucket, Key=file_name)
status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

if status == 200:
    print(f"Successful S3 get_object response. Status - {status}")
    reader1, reader2, reader3 = tee(pd.read_csv(response.get("Body"), chunksize=100), 3)
    scaler = StandardScaler()
    sklearn_pca = IncrementalPCA(n_components=1)
    for chunk in reader1:
        # scaler.partial_fit(chunk.sample(frac=0.5, random_state=42)['x'].to_numpy().reshape(1,-1))
        scaler.partial_fit(chunk.sample(frac=0.5, random_state=42)[['x','y']])
        # print(scaler.mean_, scaler.var_)

    all_scaled_data = []
    # reader = pd.read_csv(response.get("Body"), chunksize=100)
    for chunk in reader2:
        scaled_data = scaler.transform(chunk.sample(frac=0.5, random_state=42)[['x','y']])
        #add each chunk of transformed data to list
        all_scaled_data.append(scaled_data)
        sklearn_pca.partial_fit(scaled_data)
    print(len(all_scaled_data))
    # print(type(all_scaled_data))
    print(all_scaled_data[0].shape)

    pca_transformed = None
    for chunk in reader3:
        tx_chunk = sklearn_pca.transform(chunk.sample(frac=0.5, random_state=42)[['x','y']])
        if pca_transformed is None:
            pca_transformed = tx_chunk
        else:
            pca_transformed = np.vstack((pca_transformed, tx_chunk))

    print(pca_transformed.shape)
    print(sklearn_pca.explained_variance_, sklearn_pca.explained_variance_ratio_)

else:
    print(f"Unsuccessful S3 get_object response. Status - {status}")
