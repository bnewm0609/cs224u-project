import torch
import torch.nn as nn
import numpy as np
from monroe_data import MonroeData, MonroeDataEntry, Color # last two for reading pkl file
import caption_featurizers 
from color_featurizers import ColorFeaturizer, color_phi_fourier
from models import LiteralListener, LiteralSpeaker, ImaginativeListener, CaptionEncoder, CaptionGenerator, ColorGenerator, ColorSelector, ColorOnlyBaseline, LiteralSpeakerScorer
from evaluation import score_model, delta_e_dist
from experiment import FeatureHandler, evaluate_model
import argparse

################################################################
# This file contains examples of how to run experiments. It also
# catalogues most of the ones I have run so far

# 0. Common features to all the experiments
# -----------------------------------------
# 0.1 Data - all the same data is used in each of these experiments
train_data = None # MonroeData("data/csv/train_corpus_monroe.csv", "data/entries/train_entries_monroe.pkl")
dev_data = None # MonroeData("data/csv/dev_corpus_monroe.csv", "data/entries/dev_entries_monroe.pkl")
dev_data_synth = None
# write a function to do this so it takes less time to debug argparse stuff
def load_data(prefix=False):
    global train_data, dev_data, dev_data_synth
    if prefix:
        prefix = "../"
    else:
        prefix = ""
    train_data = MonroeData(prefix + "data/csv/train_corpus_monroe.csv", prefix + "data/entries/train_entries_monroe.pkl")
    dev_data = MonroeData(prefix + "data/csv/dev_corpus_monroe.csv", prefix + "data/entries/dev_entries_monroe.pkl")
    dev_data_synth  = MonroeData(prefix + "data/csv/dev_corpus_synth_10fold.csv", prefix + "data/entries/dev_corpus_synth_10fold.pkl")


# 1. Literal Listener
# -----------------------------------------
def literal_listener_experiment(train=False, evaluate=True, epochs=5, embed_dim = 100, hidden_dim = 100, color_dim= 54, model_file="model/literal_listener_5epoch_endings_tkn.params"):
        
    # Initializing featurizers
    print("Initializing featurizers")
    caption_phi = caption_featurizers.CaptionFeaturizer(tokenizer=caption_featurizers.EndingTokenizer) # Use with parameter files that end in `endings_tkn`
    # caption_phi = caption_featurizers.CaptionFeaturizer(tokenizer=caption_featurizers.WhitespaceTokenizer) # Use with parameter files don't
    color_phi = ColorFeaturizer(color_phi_fourier, "rgb", normalized=True)
    feature_handler = FeatureHandler(train_data, dev_data, caption_phi, color_phi) # target function is initialized by default

    print("Obtaining training features") # get features even if you're runnning the pretrained model for example 
    train_features = feature_handler.train_features()
    train_targets = feature_handler.train_targets()

    print("Initializing model")
    model = LiteralListener(CaptionEncoder, num_epochs = epochs)
    model.init_model(embed_dim = embed_dim, hidden_dim = hidden_dim, vocab_size = feature_handler.caption_featurizer.caption_indexer.size,
                 color_dim = color_dim)

    # to train: (probably takes about 15 min - 2 hrs) depending on # of epochs (5 - 30)
    if train:
        print("Training model and saving to {}:".format(model_file))
        model.fit(train_features, train_targets)
        model.save_model(model_file)
    else:
        print("Loading pretrained model")
        model.load_model(model_file)

    if evaluate:
        # convert the model output to a score for that particular round
        print("Evaluating model")
        output_to_score = lambda model_outputs, targets: np.exp(model_outputs[np.arange(len(model_outputs)), targets]) # get the model's predicted probablity at each target index and use that as the score
        evaluate_model(dev_data, feature_handler, model, output_to_score, score_model)
    return model

# 2. Literal Listener (trained with listener selections as target)
# -----------------------------------------
#    Everything is the same other than the target function
def literal_listener_listener_click_experiment(train=False, evaluate=True, epochs=5, embed_dim = 100, hidden_dim = 100, color_dim= 54, model_file="model/literal_listener_listener_click_5epoch_endings_tkn.params"):
    
    # Initializing featurizers
    print("Initializing featurizers")
    caption_phi = caption_featurizers.CaptionFeaturizer(tokenizer=caption_featurizers.EndingTokenizer) # Use with parameter files that end in `endings_tkn` - using endings tokenizer to separate endings like "ish" and "er"
    # caption_phi = caption_featurizers.CaptionFeaturizer(tokenizer=caption_featurizers.WhitespaceTokenizer) # Use with other paremter files
     
    color_phi = ColorFeaturizer(color_phi_fourier, "rgb", normalized=True)

    # Now we have a different target function, because we want to predict what the listener selected
    def listener_click_target(data_entry, color_perm):
        # color_perm because colors are randomized
        return np.where(color_perm==data_entry.click_idx)[0]

    feature_handler = FeatureHandler(train_data, dev_data, caption_phi, color_phi, target_fn=listener_click_target) 

    print("Obtaining training features") # get features even if you're runnning the pretrained model for example
    train_features = feature_handler.train_features()
    train_targets = feature_handler.train_targets()

    print("Initializing model")
    model = LiteralListener(CaptionEncoder, num_epochs = epochs)
    model.init_model(embed_dim = embed_dim, hidden_dim = hidden_dim, vocab_size = feature_handler.caption_featurizer.caption_indexer.size,
                 color_dim = color_dim)

    # to train: (probably takes about 15 min - 2 hrs) depending on # of epochs (5 - 30)
    if train:
        print("Training model and saving to {}:".format(model_file))
        model.fit(train_features, train_targets)
        model.save_model(model_file)
    else:
        print("Loading pretrained model")
        model.load_model(model_file)

    if evaluate:
        # convert the model output to a score for that particular round
        print("Evaluating model")
        output_to_score = lambda model_outputs, targets: np.exp(model_outputs[np.arange(len(model_outputs)), targets]) # get the model's predicted probablity at each target index and use that as the score
        # we want to score based on the model's predictions at the TARGET indices not listener clicked indices, 
        # so we change the feature_handler's target function to do that:
        feature_handler.target_fn = lambda data_entry, color_perm: np.where(color_perm == data_entry.target_idx)[0]
        evaluate_model(dev_data, feature_handler, model, output_to_score, score_model)
    return model

# 3. Literal Speaker
# -----------------------------------------
def literal_speaker_experiment(train=False, evaluate=True, epochs=5, color_in_dim = 54, color_dim = 100, embed_dim = 100, hidden_dim = 100, lr = 0.004, model_file="model/literal_speaker_30epochGLOVE.params"):
    # Initializing featurizers
    print("Initializing featurizers")
    caption_phi = caption_featurizers.CaptionFeaturizer(tokenizer=caption_featurizers.EndingTokenizer)  # we'll use the EndingTokenizer in order to have common training data, but should technically be WhitespaceTokenizer

    color_phi = ColorFeaturizer(color_phi_fourier, "hsv", normalized=True) # speaker uses hsv 
    
    # speaker's target is to predict tokens following the SOS token
    def speaker_target(data_entry):
        _, caption_ids = caption_phi.to_string_features(data_entry.caption) # this probably works...
        target = caption_ids[1:]
        return target

    feature_handler = FeatureHandler(train_data, dev_data, caption_phi, color_phi, target_fn=speaker_target, randomized_colors=False)

    print("Obtaining training features")
    train_features = feature_handler.train_features()
    train_targets = feature_handler.train_targets()

    print("Initializing model")
    model = LiteralSpeaker(CaptionGenerator, optimizer=torch.optim.Adam, lr=lr, num_epochs=epochs)
    model.init_model(color_in_dim=color_in_dim, color_dim=color_dim,
                                  vocab_size=caption_phi.caption_indexer.size, embed_dim=embed_dim,
                                 speaker_hidden_dim=hidden_dim)
    if train:
        print("Training model and saving to {}:".format(model_file))
        model.fit(train_features, train_targets)
        model.save_model(model_file)
    else:
        print("Loading pretrained model")
        model.load_model(model_file)

    if evaluate:
        # do some kind of evaluation...
        print("TODO: No evaluation currently set for speaker model.")
    return model


# 4. Imaginative Listener
def imaginative_listener(train=False, model_file="model/imaginative_listener_with_distractors_linear100hd5epoch_GLOVE_MSE.params"):
    print("Initializing featurizers")
    caption_phi = caption_featurizers.CaptionFeaturizer(tokenizer=caption_featurizers.EndingTokenizer)  
    color_phi = ColorFeaturizer(color_phi_fourier, "rgb", normalized=True) 

    def target_color_target(data_entry):
        return np.array(data_entry.colors[0].rgb_norm)

    feature_handler = FeatureHandler(train_data, dev_data_synth, caption_phi, color_phi, target_fn=target_color_target,
                                randomized_colors=False)

    print("Obtaining training features") # get features even if you're runnning the pretrained model for example
    train_features = feature_handler.train_features()
    train_targets = feature_handler.train_targets()

    imaginative_model = ImaginativeListener(ColorGenerator, criterion=torch.nn.CosineEmbeddingLoss,
                            optimizer=torch.optim.Adam, lr=0.004, num_epochs=5)

    MSELossSum = lambda: nn.MSELoss(reduction='sum') # sorry for this ugliness..... but this is me passing a parameter to the loss func
    imaginative_model = ImaginativeListener(ColorGenerator, criterion=MSELossSum,
                                optimizer=torch.optim.Adam, lr=0.004, num_epochs=5, use_color=True)
    imaginative_model.init_model(embed_dim=100, hidden_dim=100, vocab_size=feature_handler.caption_featurizer.caption_indexer.size,
                    color_in_dim=54, color_hidden_dim=100, weight_matrix=caption_featurizers.get_pretrained_glove(feature_handler.caption_featurizer.caption_indexer.idx2word.items(), 100))

    if train:
        print("Training model and saving to {}:".format(model_file))
        imaginative_model.fit(train_features, train_targets)
        imaginative_model.save_model(model_file)
    else:
        print("Loading pretrained model")
        imaginative_model.load_model(model_file)

    print("Evaluating model")
    output_to_score_de = lambda outputs, targets: np.array([delta_e_dist(outputs[i], targets[i]) for i in range(len(targets))])
    # we want to score based on the model's predictions at the TARGET indices not listener clicked indices, 
    # so we change the feature_handler's target function to do that:
    evaluate_model(dev_data_synth, feature_handler, imaginative_model, output_to_score_de, score_model, accuracy=False)


#5. Color-Only Baseline
def color_only_baseline(train=False, model_file="model/baseline_model.params"):
    caption_phi = caption_featurizers.CaptionFeaturizer(tokenizer=caption_featurizers.EndingTokenizer) # Use with parameter files that end in `endings_tkn` - using endings tokenizer to separate endings like "ish" and "er"
    color_phi = ColorFeaturizer(color_phi_fourier, "rgb", normalized=True)
    feature_handler = FeatureHandler(train_data, dev_data_synth, caption_phi, color_phi)
    train_features = feature_handler.train_features()
    train_targets = feature_handler.train_targets()

    baseline_model = ColorOnlyBaseline(ColorSelector, optimizer=torch.optim.Adam, lr=0.001, num_epochs=5)
    baseline_model.init_model(color_dim=54)

    if train:
        print("Training model and saving to {}.".format(model_file))
        baseline_model.fit(train_features, train_targets)
        baseline_model.save_model(model_file)
    else:
        print("Loading pretrained model")
        baseline_model.load_model(model_file)

    print("Evaluating model")
    output_to_score = lambda model_outputs, targets: np.exp(model_outputs[np.arange(len(model_outputs)), targets]) # get the model's predicted probablity at each target index and use that as the score
    evaluate_model(dev_data_synth, feature_handler, baseline_model, output_to_score, score_model)

# 6. Rebuen & Will's LM scorer
def literal_speaker_scorer(train=False, model_file="model/literal_speaker_30epochGLOVE.params"):
    caption_phi = caption_featurizers.CaptionFeaturizer(tokenizer=caption_featurizers.EndingTokenizer) 
    color_phi = ColorFeaturizer(color_phi_fourier, "hsv", normalized=True) # speaker uses hsv

    # speaker's target is to predict tokens following the SOS token
    def speaker_target(data_entry):
        _, caption_ids = caption_phi.to_string_features(data_entry.caption) # this probably works...
        target = caption_ids[1:]
        return target

    feature_handler = FeatureHandler(train_data, dev_data_synth, caption_phi, color_phi, target_fn=speaker_target, randomized_colors=False)

    if train:
        print("Obtaining training features")
        train_features = feature_handler.train_features()
        train_targets = feature_handler.train_targets()


    lss_model = LiteralSpeakerScorer(CaptionGenerator)
    lss_model.init_model(color_in_dim=54, color_dim=100,
                                  vocab_size=caption_phi.caption_indexer.size, embed_dim=100,
                                 speaker_hidden_dim=100)

    if train:
        pass
    else:
        lss_model.load_model(model_file)

    # unused, but here for reference
    def output_to_score_lss(outputs, targets):
        """ 
        Scoring function for listener scorer. Model outputs are probabilities of sentences
        under the conditional language model. To get the score, we see if the plurality of the
        probability mass is on the target, and if so the score is 1. Otherwise the score is 0
        """
        all_scores = []
        for i, predictions in enumerate(outputs):
            scores = [0, 0, 0]
            for j, prediction in enumerate(predictions):
                scores[j] = np.sum(prediction[np.arange(len(targets[i])), targets[i]].numpy())
            all_scores.append(scores)
        return np.argmax(np.array(all_scores), axis=1) == 0 # all the targets are at index 0


    def output_to_score_lss_probmass(outputs, targets):
        """ 
        Scoring function for listener scorer. The model outputs are probabilities of sentences
        under the conditional language model. To get the score, we softmax those probabilities 
        and take the probability mass of the target color as the model score.
        """
        all_scores = []
        for i, predictions in enumerate(outputs):
            scores = np.array([0, 0, 0], dtype=np.float64)
            for j, prediction in enumerate(predictions):
                scores[j] = np.sum(prediction[np.arange(len(targets[i])), targets[i]].numpy()) # markov assumption to sum log probabilities of words
            # softmax scores
            scores = np.exp(scores) / np.sum(np.exp(scores))
            # take the portion of the distribution asssigned to target (@ index 0)
            all_scores.append(scores[0])
        return all_scores

    result = evaluate_model(dev_data_synth, feature_handler, lss_model, output_to_score_lss, score_model, accuracy=False)
    print(result)




if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=['literal_listener', 'literal_speaker', 'literal_listener_listener_click', 'imaginative_listener',
                                            'color_only_baseline', 'literal_speaker_scorer'],
        help="Which model you want to run")
    parser.add_argument("--retrain", default=False, help="Set to true if you want to retrain the model")
    parser.add_argument("--model_file", default=None, help="Set to load pretrained or save trained model")

    args = parser.parse_args()

    if args.model == 'literal_listener':
        experiment_func = literal_listener_experiment
    elif args.model == 'literal_speaker':
        experiment_func = literal_speaker_experiment
    elif args.model == 'literal_listener_listener_click':
        experiment_func = literal_listener_listener_click_experiment
    elif args.model == 'imaginative_listener':
        experiment_func = imaginative_listener
    elif args.model == 'color_only_baseline':
        experiment_func = color_only_baseline
    elif args.model == 'literal_speaker_scorer':
        experiment_func = literal_speaker_scorer
    else:
        print("Model not recognized")

    load_data()

    if args.model_file is None:
        experiment_func() # don't retrain and overwrite the default model
    else:
        experiment_func(train = args.retrain, model_file = args.model_file)


