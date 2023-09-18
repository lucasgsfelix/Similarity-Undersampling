import os

import pandas as pd

import numpy as np

import tqdm

from sklearn.linear_model import LogisticRegression

import texthero as hero

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

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

	similarities = []

	for file_name in tqdm.tqdm(os.listdir("Similarites")):

		similarities.append(pd.read_parquet("Similarites/" + file_name))

	similarity_df = pd.concat(similarities)

	similarities = []

	df = df[['text', 'trip type', 'dataset', 'review_clean']]

	parameters, complete_results = [], []

	model_parameters = tested_parameters("LogisticRegression")

	train_dataset = "TripAdvisor"

	for similarity in [True, False]:

		# se a similaridade for == True, então quer dizer temos aqueles que são menos similares
		# se a similaridade for == False, então quer dizer que temos aqueles que são mais similares

		similarity_df.sort_values(by='similarity', ascending=similarity, inplace=True)

		print("Fim da ordenação de similaridades!")

		for amount_instances in tqdm.tqdm([1000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000,
						   150000, 200000, 250000, 300000,
						   350000, 400000, 450000, 500000, 550000, len(df)]):

			amount_work_instances = int(np.ceil(amount_instances * tripadvisor_info['Work']))

			amount_leisure_instances = int(np.ceil(amount_instances * tripadvisor_info['Leisure']))

			## mantendo a mesma distribuição
			work_instances = similarity_df[similarity_df['class_trip'] == 1].drop_duplicates(subset='doc_trip', keep='first')

			work_instances = work_instances.head(amount_work_instances)['doc_trip'].values

			leisure_instances = similarity_df[similarity_df['class_trip'] == 0].drop_duplicates(subset='doc_trip', keep='first')

			leisure_instances = leisure_instances.head(amount_leisure_instances)['doc_trip'].values

			selected_instances = df[(((df['dataset'] == 'TripAdvisor')) &
						((df.index.isin(work_instances)) | (df.index.isin(leisure_instances)))) |
						(df['dataset'] == 'Yelp')]

			selected_instances.sort_values(by='dataset', ascending=False, inplace=True)
			# treinando modelo
			pipe = Pipeline(steps=[("model", LogisticRegression())])

			vectorizer = TfidfVectorizer()

			X = vectorizer.fit_transform(selected_instances['review_clean'])

			size_tripadvisor = amount_work_instances + amount_leisure_instances

			x_train, y_train = X[-size_tripadvisor:], selected_instances['trip type'][-size_tripadvisor: ]

			x_test, y_test = X[: size_yelp], selected_instances['trip type'][: size_yelp]

			grid = GridSearchCV(estimator=pipe,
			param_grid=[model_parameters],
			cv=KFold(n_splits=5),
			scoring=('f1_micro', 'f1_macro'),
			return_train_score=True,
			n_jobs=-1,
			error_score=0,
			verbose=0,
			refit=False)

			grid.fit(x_train, y_train)

			grid_results = pd.DataFrame(grid.cv_results_)

			best_parameters = retrieve_best_params(grid_results)

			trained_model = LogisticRegression(**best_parameters).fit(x_train, y_train)

			prediction = trained_model.predict(x_test)

			results = {
				'Model': 'LogisticRegression',
				'f1-micro': f1_score(y_test, prediction, average='micro'),
				'f1-macro': f1_score(y_test, prediction, average='macro'),
				'f1-binary': f1_score(y_test, prediction, average='binary'), 
				'accuracy': accuracy_score(y_test, prediction),
				'less_similar': similarity,
				"amount_instances": amount_instances
			}

			complete_results.append(pd.DataFrame([results]))

			parameters.append(grid_results)

	df_complete = pd.concat(complete_results)

	df_parameters = pd.concat(parameters)

	df_complete.to_csv(train_dataset + ".csv", sep=';', index=False)

	df_parameters.to_csv(train_dataset + "_parameters.csv", sep=';', index=False)

