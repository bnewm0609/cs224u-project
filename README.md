# Communication-based Evaluation for Natural Language Generation

# Project Description 
Currently many NLG models are evaluated using n-gram overlap metrics like BLEU and ROUGE, but these metrics do not capture semantics let alone speaker intentions. People use language to communicate, and if we want NLG models to effectively communicate with people, we should evaluate them based on how well they communicate. We illustrate how this communication-based evaluation would work using the color reference game scenario from Monroe et al., 2017. We collected color reference game captions of various qualities and investigated how well models that use the captions to play the reference game can distinguish between dffierent quality captions compared to n-gram overlap metrics.

We have released the data we collected. It can be found in [data/csv/clean_data.csv](./data/csv/clean_data.csv). The code to recreate the plots in the paper using the data and pretrained models can be found in this [jupyter notebook](./notebooks/Replication%20Example%20Notebook.ipynb).

# Folder and File Descriptions
[caption_featurizers.py](./caption_featurizers.py) contains code to process captions with an appropriate tokenizer into a format expected by the models. [color_featurizers.py](./color_featurizers.py) is a similar featurizer for the color inputs.

[evaluation.py](./evaluation.py) contains performance metric code for all models.

[example_experiments.py](./example_experiments) contains examples of experiments that can be run with models such as the Literal Listener.

[experiment.py](./experiment.py) contains code for model evaluation and the feature handler class that interfaces between the Monroe data, feature functions, and the models.

[baseline_listener_samples](./baseline_listener_samples/), [literal_listener_samples](./literal_listener_samples/), and [imaginative_listener_samples](./imaginative_listener_samples/) contain the ten sampled model parameters with optimal hyperparameters from the Baseline, Literal, and Imaginative Listener models, respectively.

[data](./data/) contains all the data used in the project, including the Monroe data and the synthetic data.

[model](./model/) contains all other model parameters for the models experimented with over the course of the project.

[notebooks](./notebooks/) contains Jupyter notebooks for the experiments and scripts used to explore data, generate models, run models, sample models, score models, and other tasks.

