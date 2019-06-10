# CS224U Project
Much of the code for this project has been based off of the code at the [futurulus/colors-in-context](https://github.com/futurulus/colors-in-context) GitHub.

# Project Abstract
In this project, we present a framework for evaluating natural language descriptions in the color captioning problem. In this task, two agents are given a set of three colors and one of them generates a description of a target color for the other agent.  Our approach is pragmatically motivated:  we measure the effectiveness of a caption in terms of how well a trained model can select the correct color given the caption.  We investigate four models, two of which explicitly model pragmatic reasoning, and we formulate a performance metric based on Gricean maxims to compare the effectiveness of the models.  Our results indicate that though modeling pragmatic reasoning explicitly does improve evaluation performance by a small margin, it may not be essential from a practical perspective. Overall, we believe this evaluation framework is a promising start for evaluating natural language descriptions of captioning systems.

# Folder and File Descriptions
[caption_featurizers.py](./caption_featurizers.py) contains code to process captions with an appropriate tokenizer into a format expected by the models. [color_featurizers.py](./color_featurizers.py) is a similar featurizer for the color inputs.

[evaluation.py](./evaluation.py) contains performance metric code for all models.

[example_experiments.py](./example_experiments) contains examples of experiments that can be run with models such as the Literal Listener.

[experiment.py](./experiment.py) contains code for model evaluation and the feature handler class that interfaces between the Monroe data, feature functions, and the models.

[synthetic_data.py](./synthetic_data.py) contains scripts for generating the synthetic data used in the project that had more balanced classes of correctly and incorrectly identified colors than the original Monroe dataset (with a one-to-one ratio of correct and incorrect listener responses).

[baseline_listener_samples](./baseline_listener_samples/), [literal_listener_samples](./literal_listener_samples/), and [imaginative_listener_samples](./imaginative_listener_samples/) contain the ten sampled model parameters with optimal hyperparameters from the Baseline, Literal, and Imaginative Listener models, respectively.

[data](./data/) contains all the data used in the project, including the Monroe data and the synthetic data.

[model](./model/) contains all other model parameters for the models experimented with over the course of the project.

[notebooks](./notebooks/) contains Jupyter notebooks for the experiments and scripts used to explore data, generate models, run models, sample models, score models, and other tasks.

[results](./results/) contains final model performance metric results for the Literal, Imaginative, and Pragmatic Listeners.
