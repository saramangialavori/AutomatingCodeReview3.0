# Automating Code Review 3.0

This repository consists in the replication package of the thesis **"Automating Code Review: Using Deep Learning to Assess the Quality of Code Contributions and Recommend Changes"**.

The thesis is based on the work of Tufano *et al.* **"Using Pre-trained Models to Boost Code Review Automation"**. In particular, we trained the model on the same code review tasks using different datasets that include additional contextual information regarding either the change diff of the current review round or the past conversation between the contributor and the reviewer. Moreover, we trained a classifier able to determine whether a given piece of code is of high- or low-quality.

In this repository you will find all the resources needed to replicate our work.

## Resources

The code is split as follows:

- In the `manual_inspection` folder you will find the code needed to run the heuristics and random forest classifier we implemented to filter the comments 
- The `replication` folder contains the code to replicate the work of Tufano *et al.* by cleaning the dataset and creating the train/validation/test sets for each code review task
- The `diff_context` folder contains the code to add the change diff context of the current review round to the instances and creating the corresponding train/validation/test sets for each code review task
- The `conversation_context` folder contains the code to add the past reviewers-contributor conversation to the instances and creating the corresponding train/validation/test sets for each code review task
- In the `classifier` folder you can find the code to add the merging information to the instances and creating the train/validation/test sets to give as input to the classifier
- The folder `evaluation` contains the code to compute the accuracy in terms of perfect predictions as we did to evaluate our models
- The `tokenizer` folder contains the tokenizer used by our models
- In the `notebooks` folder you will find the Google Colab notebooks we used for the finetuning of the models
- The folder `utils` contains useful resources used when cleaning the data or adding contextual information to the instances
- `requirements.txt` lists the Python packages needed to run our code

At this [link](https://zenodo.org/record/6609052#.YpneBi8Rpc9) you can find extra materials that are needed to fully replicate our experiments:

- `datasets.zip` contains all the processed and split datasets we used
	- replication
		- code-to-code 
			- `train.tsv`, `val.tsv`, `test.tsv`
		- code&comment-to-code
			- `train.tsv`, `val.tsv`, `test.tsv`
		- code-to-comment
			- `train.tsv`, `val.tsv`, `test.tsv`
	- replication\_by\_time
		- code-to-code 
			- `train.tsv`, `val.tsv`, `test.tsv`
		- code&comment-to-code
			- `train.tsv`, `val.tsv`, `test.tsv`
		- code-to-comment
			- `train.tsv`, `val.tsv`, `test.tsv`
	- diff_context
		- code-to-code 
			- `train.tsv`, `val.tsv`, `test.tsv`
		- code&comment-to-code
			- `train.tsv`, `val.tsv`, `test.tsv`
		- code-to-comment
			- `train.tsv`, `val.tsv`, `test.tsv`
	- conversation_context
		- code-to-code 
			- `train.tsv`, `val.tsv`, `test.tsv`
		- code&comment-to-code
			- `train.tsv`, `val.tsv`, `test.tsv`
		- code-to-comment
			- `train.tsv`, `val.tsv`, `test.tsv`
	- classifier
		- `train.tsv`, `val.tsv`, `test.tsv`

- `models.zip` contains the best checkpoints of the models we trained

- `results.zip` includes the inputs, targets and predictions to evaluate the model on the test set. For each combination of model and task we have the following files
	- `inputs.txt`: input file for the model
	- `targets.txt`: expected output of the model
	- `predictions.txt`: predictions generated by the model
	- `confidence_scores.txt`: confidence scores of the corresponding predictions

The dataset with the original data is available upon request due to its size.

## Manual inspection
- `manual_analysis_all.csv`: contains the manually inspected data of the unfiltered instances
- `manual_analysis_filtered.csv`: contains the manually inspected data of the surviving instances after the filtering process
- `manual_analysis_filtered_complete.csv`: contains all the data related to the surviving instances after the filtering process (hence, the whole conversation related to a surviving comment)

In order to run the following files, you need to download the code of the sentiment analysis tools [SentiStrength-SE](http://sentistrength.wlv.ac.uk) and [SentiCR](https://github.com/senticr/SentiCR) and place them in the folder `manual_inspection`.

- `build_oracle.py`: is used to build the oracle with the manually analyzed data of the CSV files
- `get_statistics.py`: the output shows the statistics regarding the last answer of a conversation
- `heuristics.py`: runs the evaluation of heuristics we developed by optionally selecting a sentiment analysis tool between "sentistrength" and "senticr" with the option `-p`
- `random_forest_classifier.py`: runs the evaluation of the random forest classifier. You can use option `-s` to change the strategy ("last_answer" or "answers"), `-p` to change the sentiment analysis tool, and `-b` to balance the data

## Replication
In this and the next steps we expect the original raw data (available upon request) to be in a folder called `data`.

- `Analyzer.py`, `Cleaner.py`: the two main classes to preprocess the dataset. Search for a `#TODO` in the `Cleaner.py` to insert [your JSON token](https://cloud.google.com/translate/docs/setup) if you want to employ the [Google language detection library](https://cloud.google.com/translate/docs/basic/detecting-language).
- `analyze_data.py`: filters and cleans the data by providing the path to the folder containing the data, expected as CSV files
- `create_datasets.py`: randomly splits the processed data into the train/validation/test sets for each of the three code review tasks
- `create_datasets_split_by_time.py`: splits the processed data into the train/validation/test sets for each of the three code review tasks, by considering the creation date of the comment

## Diff context
- `extract_diff_context.py`: adds the change diff context to the processed instances of the dataset
- `create_datasets_with_diff_context.py`: splits by time the processed data into the train/validation/test sets for each of the three code review tasks, adding the change diff context

## Conversation context
- `extract_conversations.py`: adds the context of the past reviewers-contributor conversation to the processed instances of the dataset
- `create_datasets_with_conversation_context.py`: splits by time the processed data into the train/validation/test sets for each of the three code review tasks, adding the context of the past conversation

## Classifier
- `check_methods_merged.py`: annotates the instances whose revised version was merged at the end of the pull request
- `create_classifier_instances.py`: generates the artificial instances representing high-quality code and splits the data into the train/validation/test sets

## Evaluation
- `compute_accuracy`: computes the accuracy in terms of perfect prediction. Requires the path to a folder containing the inputs, targets and predictions of a model checkpoint.

## Tokenizer 
- `TokenizerModel.model`: the Sentencepiece tokenizer used by our models
- `TokenizerModel.vocab`: the extracted vocabulary obtained by the training of the pre-trained model of Tufano *et al.*

## Notebooks 
- `FineTuning512.ipynb`: notebook used for the training in the replication step, with a token length equal to 512
- `FineTuning1024.ipynb`: notebook used for the training of the models with context, with a token length equal to 1024 and additional custom tags added to the tokenizer
- `FineTuningClassifier.ipynb`: notebook used for the training of the classifier

## Utils
- `github_requests.py`: contains utility functions to perform requests to the GitHub API. Requires to set the GitHub username and token.
- `stopwords.py`: contains an utility function to get a custom set of stop words used while processing comments when cleaning the data and when adding the conversation context.
- `stop-words-english.txt`: the list of English stop words
- `my_idioms_300.txt`: a list of custom idioms
