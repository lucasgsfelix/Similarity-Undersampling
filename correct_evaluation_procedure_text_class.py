import os

import pandas as pd

import numpy as np

import tqdm

from sklearn.linear_model import LogisticRegression

import texthero as hero

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import f1_score, accuracy_score


def retrieve_best_params(grid_results):

    params = grid_results[grid_results['mean_test_f1_macro'] == grid_results['mean_test_f1_macro'].max()]['params'].values[0]

    best_params = {}

    # os parâmetros dos modelos começam com 'model__'
    for key, value in params.items():

        best_params[str(key).replace('model__', '')] = value

    return best_params


def tested_parameters(model):

	return {
	"LogisticRegression": {
	#'model__penalty': ['none', 'l2'],
	#'model__C': [0.00001, 0.001, 0.1, 0.5, 1, 2, 10],
	#'model__tol': [1e-6, 1e-4, 1e-2, 1, 2, 10],
	#'model__fit_intercept': [True, False],
	#'model__solver': ['sag', 'saga'],
	#'model__max_iter': [10, 50, 100, 150, 1000, 2000, 5000, 10000],
	'model__n_jobs': [-1],
	'model__class_weight': ['balanced', None]
	}

	}[model]



def retrieve_train_data(x_train, y_train, func_used, amount_instances):

	if func_used == 'head':

		func_used_string = 'Menos Similares'

		# fold x train selected
		fxts, fyts = fold_x_train.head(amount_instances), fold_y_train.head(amount_instances)

	else:

		func_used_string = 'Mais Similares'

		fxts, fyts = fold_x_train.tail(amount_instances), fold_y_train.tail(amount_instances)

	return fxts, fyts, func_used_string

if __name__ == '__main__':


	df = pd.read_table("trip_advisor_dataset.csv", sep=';')

	df_yelp = pd.read_table("manual_reviews.csv", sep=';')

	df['dataset'] = 'TripAdvisor'

	df_yelp['dataset'] = 'Yelp'

	size_yelp, size_tripadvisor = len(df_yelp), len(df)

	df = pd.concat([df_yelp, df]).reset_index(drop=True)


	tripadvisor_info = {
				"Work": len(df[(df['trip type'] == 1) & (df['dataset'] == 'TripAdvisor')])/len(df),
				"Leisure": len(df[(df['trip type'] == 0) & (df['dataset'] == 'TripAdvisor')])/len(df)
				}

	df['review_clean'] = hero.clean(df['text'])

	# primeiro Yelp e depois TripAdvisor
	df = df[['text', 'trip type', 'dataset', 'review_clean']]

	similarities = []

	for file_name in tqdm.tqdm(os.listdir("Similarites")):

		similarities.append(pd.read_parquet("Similarites/" + file_name))

	similarity_df = pd.concat(similarities)

	similarities = []

	similarity_df.sort_values(by='similarity', ascending=True, inplace=True)

	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(df['review_clean'])

	## dados do trip advisor
	x_train, y_train = X[-size_tripadvisor:], df['trip type'][-size_tripadvisor: ]

	## dados do yelp
	#x_test, y_test = X[: size_yelp], df['trip type'][: size_yelp]

	kfolds = StratifiedKFold(5)

	folds_index = {fold: {'train': train_index, 'test': test_index}
				  for fold, (train_index, test_index) in enumerate(kfolds.split(x_train, y_train))}


	# se a similaridade for == True, então quer dizer temos aqueles que são menos similares
	# se a similaridade for == False, então quer dizer que temos aqueles que são mais similares

	complete_results = []

	for fold in tqdm.tqdm(folds_index.keys()):

		fold_test_index = folds_index[fold]['test']

		# if head is the less similar, if tail is the most similar
		for func_used in ['head', 'tail']:

			for amount_instances in tqdm.tqdm([1000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000,
											   150000, 200000, 250000, 300000,
											   350000, 400000, 450000, len(folds_index[fold]['train'])]):


				if func_used == 'head':

					func_used_string = 'Menos Similares'

					similarity_used = similarity_df.drop_duplicates(subset='doc_trip', keep='first')

					# eu quero as X primeiro instâncias que não estão no teste
					index_train = similarity_used[~similarity_used['doc_trip'].isin(fold_test_index)]['doc_trip'].head(amount_instances).values

					fxts = x_train[index_train]

					fyts = y_train[index_train]

				else:

					func_used_string = 'Mais Similares'

					similarity_used = similarity_df.drop_duplicates(subset='doc_trip', keep='last')

					index_train = similarity_used[~similarity_used['doc_trip'].isin(fold_test_index)]['doc_trip'].tail(amount_instances).values

				fxts = x_train[index_train]

				fyts = y_train[index_train]

				y_test_fold = y_train[fold_test_index]

				for balanced in [True, False]:

					model = LogisticRegression(balanced=balanced).fit(fxts, fyts)

					# estamos fazendo isso apenas sobre a base de dados do tripadvisor
					prediction = model.predict(x_train[fold_test_index])


					results = {
						'Model': 'LogisticRegression',
						'f1-micro-trip': f1_score(y_test_fold, prediction, average='micro'),
						'f1-macro-trip': f1_score(y_test_fold, prediction, average='macro'),
						'order_by': func_used_string,
						'model_train_balancing': balanced,
						'fold': fold,
						"amount_instances_train": amount_instances,
						"test_base": "TripAdvisor"
					}

					complete_results.append(results)


	df_complete = pd.concat(complete_results)

	df_complete.to_csv("train_test_tripadvisor.csv", sep=';', index=False)
