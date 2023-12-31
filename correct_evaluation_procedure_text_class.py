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

	df = pd.concat([df, df_yelp]).reset_index(drop=True)

	#df = df.reset_index()


	tripadvisor_info = {
				"Work": len(df[(df['trip type'] == 1) & (df['dataset'] == 'TripAdvisor')])/len(df),
				"Leisure": len(df[(df['trip type'] == 0) & (df['dataset'] == 'TripAdvisor')])/len(df)
				}

	df['review_clean'] = hero.clean(df['text'])

	# primeiro Yelp e depois TripAdvisor
	df = df[['text', 'trip type', 'dataset', 'review_clean']]

	similarities = []
	
	if "similaridade_df.csv" not in os.listdir():

		for file_name in tqdm.tqdm(os.listdir("Similarites")):

			similarities.append(pd.read_parquet("Similarites/" + file_name))

		similarity_df = pd.concat(similarities)

		similarities = []

		similarity_df.sort_values(by='similarity', ascending=True, inplace=True)

		similarity_df.to_csv("similaridade_df.csv", sep=';', index=False)

	else:
	
		similarity_df = pd.read_table("similaridade_df.csv", sep=';')

	vectorizer = TfidfVectorizer()

	# estamos utilizando apenas o tripadvisor
	vectorizer.fit(df['review_clean'])

	## dados do trip advisor
	x_train, y_train = vectorizer.transform(df[df['dataset'] == 'TripAdvisor']['review_clean'].values),  df[df['dataset'] == 'TripAdvisor']['trip type'].values

	## dados do yelp
	x_test, y_test = vectorizer.transform(df[df['dataset'] == 'Yelp']['review_clean'].values),  df[df['dataset'] == 'Yelp']['trip type'].values

	kfolds = StratifiedKFold(5)

	folds_index = {fold: {'train': train_index, 'test': test_index}
				  for fold, (train_index, test_index) in enumerate(kfolds.split(x_train, y_train))}


	# se a similaridade for == True, então quer dizer temos aqueles que são menos similares
	# se a similaridade for == False, então quer dizer que temos aqueles que são mais similares

	complete_results, yelp_results = [], []

	for fold in tqdm.tqdm(folds_index.keys()):

		# aqui estamos trabalhando com a indexação original do dataset, por isso eu somo pelo size_yelp
		fold_test_index = folds_index[fold]['test']

		# if head is the less similar, if tail is the most similar
		for func_used in ['head', 'tail']:

			for amount_instances in [1000]:#, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000,
								#			   150000, 200000, 250000, 300000,
								#			   350000, 400000, 450000, len(folds_index[fold]['train'])]):

				amount_work_instances = int(np.ceil(amount_instances * tripadvisor_info['Work']))

				amount_leisure_instances = int(np.ceil(amount_instances * tripadvisor_info['Leisure']))

				if func_used == 'head':

					func_used_string = 'Menos Similares'

					similarity_used = similarity_df.drop_duplicates(subset='doc_trip', keep='first')

					# eu quero as X primeiro instâncias que não estão no teste
					index_train = similarity_used[~similarity_used['doc_trip'].isin(fold_test_index)]#['doc_trip'].head(amount_instances).values

					work_instances = index_train[index_train['class_trip'] == 1]['doc_trip'].head(amount_work_instances).values

					leisure_instances = index_train[index_train['class_trip'] == 0]['doc_trip'].head(amount_leisure_instances).values

					general_work_instances = similarity_used[similarity_used['class_trip'] == 1]['doc_trip'].head(amount_work_instances).values

					general_leisure_instances = similarity_used[similarity_used['class_trip'] == 0]['doc_trip'].head(amount_leisure_instances).values

					## vai retornar as X instâncias mais/menos similares no geral
					yelp_similarity = np.append(general_work_instances, general_leisure_instances)

				else:

					func_used_string = 'Mais Similares'

					similarity_used = similarity_df.drop_duplicates(subset='doc_trip', keep='last')

					index_train = similarity_used[~similarity_used['doc_trip'].isin(fold_test_index)]#['doc_trip'].head(amount_instances).values

					## isso aqui é para garantir que a distribuição tá certa
					work_instances = index_train[index_train['class_trip'] == 1]['doc_trip'].tail(amount_work_instances).values

					leisure_instances = index_train[index_train['class_trip'] == 0]['doc_trip'].tail(amount_leisure_instances).values

					general_work_instances = similarity_used[similarity_used['class_trip'] == 1]['doc_trip'].tail(amount_work_instances).values

					general_leisure_instances = similarity_used[similarity_used['class_trip'] == 0]['doc_trip'].tail(amount_leisure_instances).values

					## vai retornar as X instâncias mais/menos similares no geral
					yelp_similarity = np.append(general_work_instances, general_leisure_instances)

				# a indexação vinda da similaridade contém os dados do yelp, porém, a matriz de tf-idf não
				# por esse motivo estamos "removendo" da soma o index do yelp
												   
				index_train = np.append(work_instances, leisure_instances)

				fxts = x_train[index_train]

				fyts = y_train[index_train]

				# aqui não é a indexação original, e sim uma indexação gerada pelo kfold
				y_test_fold = y_train[fold_test_index]

				for balanced in ['balanced', None]:

					parameters = {'C': 0.001, 'fit_intercept': True, 'max_iter': 150, 
						      'n_jobs': -1, 'solver': 'saga', 'tol': 1, 'class_weight': balanced}

					model = LogisticRegression(**parameters).fit(fxts, fyts)

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

					prediction = model.predict(x_test)

					results_yelp = {
						'Model': 'LogisticRegression',
						'f1-micro-yelp': f1_score(y_test, prediction, average='micro'),
						'f1-macro-yelp': f1_score(y_test, prediction, average='macro'),
						'order_by': func_used_string,
						'model_train_balancing': balanced,
						'fold': fold,
						"amount_instances_train": amount_instances,
						"test_base": "Yelp",
						"model_geral": False
					}


					yelp_results.append(results_yelp)


					### novo modelo sendo treinado
					model = LogisticRegression(class_weight=balanced).fit(x_train[yelp_similarity], y_train[yelp_similarity])

					prediction = model.predict(x_test)

					results_yelp = {
						'Model': 'LogisticRegression',
						'f1-micro-yelp': f1_score(y_test, prediction, average='micro'),
						'f1-macro-yelp': f1_score(y_test, prediction, average='macro'),
						'order_by': func_used_string,
						'model_train_balancing': balanced,
						'fold': fold,
						"amount_instances_train": amount_instances,
						"test_base": "Yelp",
						"model_geral": True # com modelo geral queremos dizer que o modelo é treinado com as 1000 instâncias mais similares de toda a base
					}


					yelp_results.append(results_yelp)

					complete_results.append(results)


	df_complete = pd.DataFrame(complete_results)

	df_complete_yelp = pd.DataFrame(yelp_results)

	df_complete.to_csv("train_test_tripadvisor.csv", sep=';', index=False)

	df_complete_yelp.to_csv("train_trip_test_yelp.csv", sep=';', index=False)
