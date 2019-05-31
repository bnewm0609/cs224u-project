import torch
import torch.nn as nn
import numpy as np
from monroe_data import MonroeData, MonroeDataEntry, Color # last two for reading pkl file
import caption_featurizers 
from color_featurizers import ColorFeaturizer, color_phi_fourier
from models import LiteralListener, LiteralSpeaker, CaptionEncoder, CaptionGenerator
from evaluation import score_model
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
# write a function to do this so it takes less time to debug argparse stuff
def load_data():
    global train_data, dev_data
    train_data = MonroeData("data/csv/train_corpus_monroe.csv", "data/entries/train_entries_monroe.pkl")
    dev_data = MonroeData("data/csv/dev_corpus_monroe.csv", "data/entries/dev_entries_monroe.pkl")


# 1. Literal Listener
def literal_listener_experiment(train=False, model_file="model/literal_listener_5epoch_endings_tkn.params"):
    
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
    # model parameters
    embed_dim = 100; hidden_dim = 100; color_dim= 54; # hard coded for example - 54 comes from color fourier phi
    model = LiteralListener(CaptionEncoder, num_epochs=5)
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

    # convert the model output to a score for that particular round
    print("Evaluating model")
    output_to_score = lambda model_outputs, targets: np.exp(model_outputs[np.arange(len(model_outputs)), targets]) # get the model's predicted probablity at each target index and use that as the score
    evaluate_model(dev_data, feature_handler, model, output_to_score, score_model)

# 2. Literal Listener (trained with listener selections as target) 
#    Everything is the same other than the target function
def literal_listener_listener_click_experiment(train=False, model_file="model/literal_listener_listener_click_5epoch_endings_tkn.params"):
    
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
    # model parameters
    embed_dim = 100; hidden_dim = 100; color_dim= 54; # hard coded for example - 54 comes from color fourier phi
    model = LiteralListener(CaptionEncoder, num_epochs=5)
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

    # convert the model output to a score for that particular round
    print("Evaluating model")
    output_to_score = lambda model_outputs, targets: np.exp(model_outputs[np.arange(len(model_outputs)), targets]) # get the model's predicted probablity at each target index and use that as the score
    # we want to score based on the model's predictions at the TARGET indices not listener clicked indices, 
    # so we change the feature_handler's target function to do that:
    feature_handler.target_fn = lambda data_entry, color_perm: np.where(color_perm == data_entry.target_idx)[0]
    evaluate_model(dev_data, feature_handler, model, output_to_score, score_model)

# 3. Literal Speaker
def literal_speaker_experiment(train=False, model_file="model/literal_speaker_5epoch.params"):
    # Initializing featurizers
    print("Initializing featurizers")
    caption_phi = caption_featurizers.CaptionFeaturizer(tokenizer=caption_featurizers.WhitespaceTokenizer)  # use normal whitespace tokenizer (default)

    # This is the kind of feature function defined in color_featurizers.py
    # NOTE: THIS DOES NOT WORK - IT FLIPS COLORS vectors individually rather than the order of all the vectors
    # def flip_color_phi_fourier(color_list, space):
    #     """ color_list is a list of coordinates in the given color space (i.e. [1, 0, 0] for red in rgb_norm """
    #     color_features = color_phi_fourier(color_list, space)
    #     # reverse color order so target is last - this makes it so the last hidden state of
    #     # the color encoder LSTM has more recent info about the target color (shouldn't matter but that's what Monroe does)
    #     # We make a copy so it's compatible with pytorch tensors - pytorch doesn't like backwards np.arrays for some reason
    #     color_features = np.flip(color_features, axis=0).copy()
    #     return color_features

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
    speaker_model = LiteralSpeaker(CaptionGenerator, optimizer=torch.optim.Adam, lr=0.004, num_epochs=5)
    color_in_dim = 54
    color_dim = 100
    embed_dim = 100
    hidden_dim = 100
    #lit_speaker = Speaker(color_embed_dim, caption_phi.caption_indexer.size, embed_dim, hidden_dim)
    speaker_model.init_model(color_in_dim=color_in_dim, color_dim=color_dim,
                                  vocab_size=caption_phi.caption_indexer.size, embed_dim=embed_dim,
                                 speaker_hidden_dim=hidden_dim)
    if train:
        print("Training model and saving to {}:".format(model_file))
        speaker_model.fit(train_features, train_targets)
        speaker_model.save_model(model_file)
    else:
        print("Loading pretrained model")
        speaker_model.load_model(model_file)

    # do some kind of evaluation...
    print("TODO: No evaluation currently set for speaker model.")



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=['literal_listener', 'literal_speaker', 'literal_listener_listener_click'],
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

    load_data()

    if args.model_file is None:
        experiment_func() # don't retrain and save over the default model
    else:
        experiment_func(args.retrain, args.model_file)











