import pandas as pd

import texthero as hero

import numpy as np

from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer

import tqdm

from sklearn.metrics.pairwise import cosine_similarity

from multiprocessing import Pool, cpu_count

import os

def calculate_cosine_similarity(args):

	i, j, tfidf_matrix, class_trip, class_yelp = args

	vec1 = tfidf_matrix[i]

	vec2 = tfidf_matrix[j]

	similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]

	return {'doc_trip': i, 'doc_yelp': j, 'similarity': similarity,
			'class_trip': class_trip, 'class_yelp': class_yelp}


def parallel_cosine_similarity(df, tfidf_matrix, size_trip, size_yelp):

	num_docs = tfidf_matrix.shape[0]

	combinations = [(i, j, tfidf_matrix,
					df.iloc[i]['trip type'],
					df.iloc[j]['trip type'])
					for i in range(size_trip) for j in range(size_trip, (size_trip + size_yelp))]

	with Pool(cpu_count()) as pool:

		similarities = list(tqdm.tqdm(pool.imap(calculate_cosine_similarity, combinations)))

	return pd.DataFrame(similarities)



import sys


if __name__ == '__main__':

	amount_instances = int(sys.argv[1])


	if not "similarity_df.csv" in os.listdir():

		df = pd.read_table("trip_advisor_dataset.csv", sep=';')

		df_yelp = pd.read_table("manual_reviews.csv", sep=';')

		df['dataset'] = 'TripAdvisor'

		df_yelp['dataset'] = 'Yelp'


		tripadvisor_info = {
								"Work": len(df[df['trip type'] == 1])/len(df),
								"Leisure": len(df[df['trip type'] == 0])/len(df)

						   }

		df = pd.concat([df, df_yelp]).reset_index(drop=True)

		size_trip, size_yelp = len(df), len(df_yelp)

		df = df[['text', 'trip type']]

		df['review_clean'] = hero.clean(df['text'])

		vectorizer = TfidfVectorizer()

		tfidf_matrix = vectorizer.fit_transform(df['review_clean'])

		similarity_df = parallel_cosine_similarity(df, tfidf_matrix, size_trip, size_yelp)

		similarity_df = similarity_df.sort_vales(by='similarity', ascending=False)

		similarity_df.to_csv("similarity_df.csv", sep=';', index=False)

	else:

		similarity_df = pd.read_table("similarity_df.csv", sep=';')

	amount_work_instances = int(np.ceil(amount_instances * tripadvisor_info['Work']))

	amount_leisure_instances = int(np.ceil(amount_instances * tripadvisor_info['Leisure']))

	## mantendo a mesma distribuição
	work_instances = similarity_df[similarity_df['class_trip'] == 1].head(amount_work_instances)['doc_trip'].values

	leisure_instances = similarity_df[similarity_df['class_trip'] == 0].head(amount_leisure_instances)['doc_trip'].values


	selected_instances = df[(df['dataset'] == 'TripAdvisor') &
							((df.index.isin(work_instances)) | (df.index.isin(leisure_instances)))]


	selected_instances.to_csv("dataset_" + amount_instances + "_trip_advisor.csv", sep=';', index=False)
