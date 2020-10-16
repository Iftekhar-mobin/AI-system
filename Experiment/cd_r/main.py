from cd_r.retriever_sklearn import BM25Retriever
import pandas as pd

retriever = BM25Retriever(max_df=0.85)
df = pd.read_csv('/home/iftekhar/AI-system/Helpers/Title_link_merged_corpus.csv')
df['content'] = df.Article
print(df.head())

retriever.fit(df)
# print(len(retriever.features()))

best_idx_scores = retriever.predict(query='VPP')
print(best_idx_scores)