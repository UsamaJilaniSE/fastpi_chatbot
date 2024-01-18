import pandas as pd
import numpy as np
#Initialize FastAPI
from ast import literal_eval

import google.generativeai as genai

def find_best_passage(query, df_embeddings,model):
  """
  Compute the distances between the query and each document in the dataframe
  using the dot product.
  """
  query_embedding = genai.embed_content(model=model,
                                        content=query,
                                        task_type="retrieval_query")
  
  print(np.asarray(query_embedding["embedding"]).shape)
  x= np.stack(df_embeddings["Embeddings"])
  print(x.shape)
  dot_products = np.dot(np.stack(df_embeddings['Embeddings']), np.asarray(query_embedding["embedding"]))

  idx = np.argmax(dot_products)
  return df_embeddings.iloc[idx]['output_text'] # Return text from index with max value
GOOGLE_API_KEY = 'AIzaSyCyEirvh5YNKK1dekGvfw8AZf7PZN1uoko'
# df = pd.read_csv("embeddings.csv")
# embdf = df["Embeddings"]
# npemddf = np.stack(embdf)
# print(npemddf.shape)

genai.configure(api_key=GOOGLE_API_KEY)
df_embeddings = pd.read_csv('embeddings.csv')
df_embeddings.loc[:,'Embeddings'] = df_embeddings.loc[:,'Embeddings'].apply(lambda x: literal_eval(x))

query = "rury mercury wants to book a bus ticket"
model = 'models/embedding-001' 

# answer =  find_best_passage(query, df_embeddings,model)
# df_embeddings['Embeddings'] = df_embeddings['Embeddings'].apply(lambda x:x[1:-1].split(',')).apply(lambda x:[int(i) for i in x])

print(np.stack(np.asarray(df_embeddings["Embeddings"])).shape)

################ FUNCTIONS  ######################
    


