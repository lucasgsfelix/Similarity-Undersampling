import os

import pandas as pd

import numpy as np

import tqdm

if __name__ == '__main__':


	amount_instances = 10000

	df = pd.read_table("trip_advisor_dataset.csv", sep=';')

	df_yelp = pd.read_table("manual_reviews.csv", sep=';')

	df['dataset'] = 'TripAdvisor'

	df_yelp['dataset'] = 'Yelp'

	df = pd.concat([df, df_yelp]).reset_index(drop=True)

	tripadvisor_info = {
				"Work": len(df[(df['trip type'] == 1) & (df['dataset'] == 'TripAdvisor')])/len(df),
				"Leisure": len(df[(df['trip type'] == 0) & (df['dataset'] == 'TripAdvisor')])/len(df)

			   }

	similarities = []

	for file_name in tqdm.tqdm(os.listdir("Similarites")):

		similarities.append(pd.read_parquet("Similarites/" + file_name))

	similarity_df = pd.concat(similarities)

	similarities = []

	df = df[['text', 'trip type', 'dataset']]

	amount_work_instances = int(np.ceil(amount_instances * tripadvisor_info['Work']))

	amount_leisure_instances = int(np.ceil(amount_instances * tripadvisor_info['Leisure']))

	similarity_df.sort_values(by='similarity', ascending=False, inplace=True)

	## mantendo a mesma distribuição
	work_instances = similarity_df[similarity_df['class_trip'] == 1].drop_duplicates(subset='doc_trip', keep='first')

	work_instances = work_instances.head(amount_work_instances)['doc_trip'].values

	leisure_instances = similarity_df[similarity_df['class_trip'] == 0].drop_duplicates(subset='doc_trip', keep='first')

	leisure_instances = leisure_instances.head(amount_leisure_instances)['doc_trip'].values

	selected_instances = df[(df['dataset'] == 'TripAdvisor') &
							((df.index.isin(work_instances)) | (df.index.isin(leisure_instances)))]


	selected_instances.to_csv("dataset_" + str(amount_instances) + "_trip_advisor.csv", sep=';', index=False)

