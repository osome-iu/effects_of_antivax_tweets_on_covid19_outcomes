# Effects of Antivaccine Tweets on COVID-19 Vaccinations, Cases, and Deaths

Code and labeled tweets data for the paper "Effects of Antivaccine Tweets on COVID-19 Vaccinations, Cases, and Deaths" by John Bollenbacher, Fil Menczer, John Bryden. Tech Rep. [10.48550/arXiv.2406.09142](https://doi.org/10.48550/arXiv.2406.09142)



**Included code:**

* **01tweet_json_to_parquet.ipynb**.
This script cleans the raw tweet json into a more manageable tabular format with only the necessary information.

* **02clean_geodata_and_confounders.ipynb**.
This script prepares the geodata used in the analysis and some confounders data used in now-defunct model testing. 

* **03classifier_training_data_creation.ipynb**.
This notebook was used to help prepare and label tweets as Antivax or Other using Prodigy. 

* **04train_antivax_classifier.py**.
This script trained a RoBERTA-based tweet classifier over our training datasets to identify antivaccine tweets. It also contains classifier performance evaluation. 

* **05classify_geotagged_tweets.py**.
This script ran the antivax tweet classifier over our geolocated tweets dataset to determine which tweets were antivax.

* **06make_time_series_df_and_retweet_network.ipynb**.
This script transformed the raw tweets into a set of time series that were easier to analyze. It also prepares the retweet network for analysis.

* **07specify_and_run_SIRVA_model.ipynb**.
This script prepares the time series data into a clean format, specifies the SIRVA model, and runs it. It also contains a counterfactual simulation in which the number of antivaccine tweets was set to zero, as an alternative means of estimating the impact of antivaccine tweets on outcomes.

* **08causal_graphical_modeling.ipynb**.
This script specifies the causal graphical model and computes the final effect sizes.

* **09paper_plots.ipynb**.
This script contains code to produce final plots which were then prepared into our final figures in the paper.



**Included data:**

* **antivax_labels_rebalanced.parquet**.
This file contains our antivax tweet classifier training data which we labeled internally.

* **monsted_mturk_vaxx_labels.tsv**.
This file contains antivax tweet classifier training data, which Monsted et al. labeled using Mechanical Turk. This data was developed for their paper titled "Algorithmic detection and analysis of vaccine-denialist sentiment clusters in social networks" (Tech. Rep. [doi:10.48550/arXiv.1905.12908](https://doi.org/10.48550/arXiv.1905.12908)) and is available from the authors upon reasonable request.

* **test_set.parquet**.
This file contains the test set data for evaluating the classifier. This data is a random sample of our full geolocated tweets dataset, and was labeled manually by our team.



**Acknowledgments:**

* **Marissa Donofrio**, for help labeling tweet data with our team.

* **Bjarke Mønsted**, for making their labeled antivaccine tweet data available to our team in early 2021. 
