import pandas as pd

import texthero as hero

import numpy as np

from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer

import tqdm

from sklearn.metrics.pairwise import cosine_similarity

from multiprocessing import Pool, cpu_count

import os

import time

def calculate_cosine_similarity(args):

	i, j, class_trip, class_yelp = args

	vec1 = tfidf_matrix[i]

	vec2 = tfidf_matrix[j]

	similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]

	return {'doc_trip': i, 'doc_yelp': j, 'similarity': similarity,
			'class_trip': class_trip, 'class_yelp': class_yelp}


def parallel_cosine_similarity(df_trip, df_yelp):

	num_docs = tfidf_matrix.shape[0]

	# list(itertools.product(a, b))
	combinations = [(index_i, index_j,
					df_trip[(df_trip.index == index_i)]['trip type'].values[0],
					df_yelp[(df_yelp.index == index_j)]['trip type'].values[0])
					for index_i in df_trip.index for index_j in df_yelp.index]

	

	with Pool(cpu_count()) as pool:

		similarities = list(tqdm.tqdm(pool.imap(calculate_cosine_similarity, combinations)))

	return pd.DataFrame(similarities)



import sys




amount_instances = 10000

df = pd.read_table("trip_advisor_dataset.csv", sep=';')

df_yelp = pd.read_table("manual_reviews.csv", sep=';')

df['dataset'] = 'TripAdvisor'

df_yelp['dataset'] = 'Yelp'

tripadvisor_info = {
			"Work": len(df[df['trip type'] == 1])/len(df),
			"Leisure": len(df[df['trip type'] == 0])/len(df)

		   }

df = pd.concat([df, df_yelp]).reset_index(drop=True)

df = df[['text', 'trip type', 'dataset']]

if not "similarity_df.csv" in os.listdir():

	df['review_clean'] = hero.clean(df['text'])

	vectorizer = TfidfVectorizer()

	tfidf_matrix = vectorizer.fit_transform(df['review_clean'])	



	for index, partial_df in enumerate(np.array_split(df[df['dataset'] != 'Yelp'], 1000)):	

		similarity_df = parallel_cosine_similarity(partial_df, df[df['dataset'] == 'Yelp'])

		#similarity_df = similarity_df.sort_values(by='similarity', ascending=False)

		start = time.time()
		print("Escrevendo dados!")

		similarity_df.to_parquet("Similarities/similarity_df_" + str(index) + ".parquet")

		print("Finaizado escrito! ", time.time() - start)


similarity_df = pd.read_table("similarity_df.csv", sep=';')

df = df[['text', 'trip type', 'dataset']]

amount_work_instances = int(np.ceil(amount_instances * tripadvisor_info['Work']))

amount_leisure_instances = int(np.ceil(amount_instances * tripadvisor_info['Leisure']))

## mantendo a mesma distribuição
work_instances = similarity_df[similarity_df['class_trip'] == 1].head(amount_work_instances)['doc_trip'].values

leisure_instances = similarity_df[similarity_df['class_trip'] == 0].head(amount_leisure_instances)['doc_trip'].values


selected_instances = df[(df['dataset'] == 'TripAdvisor') &
						((df.index.isin(work_instances)) | (df.index.isin(leisure_instances)))]


selected_instances.to_csv("dataset_" + str(amount_instances) + "_trip_advisor.csv", sep=';', index=False)
