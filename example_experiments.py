import torch
import torch.nn as nn
import numpy as np
from monroe_data import MonroeData, MonroeDataEntry, Color # last two for reading pkl file
import caption_featurizers 
from color_featurizers import ColorFeaturizer, color_phi_fourier
import models 
from evaluation import score_model
from experiment import FeatureHandler, evaluate_model

################################################################
# This file contains examples of how to run experiments. It also
# catalogues most of the ones I have run so far

# 0. Common features to all the experiments
# -----------------------------------------
# 0.1 Data - all the same data is used in each of these experiments
train_data = MonroeData("data/csv/train_corpus_monroe.csv", "data/entries/train_entries_monroe.pkl")
dev_data = MonroeData("data/csv/dev_corpus_monroe.csv", "data/entries/dev_entries_monroe.pkl")


# 1. Literal Listener
def literal_listener_experiment(train=False, model_file="model/literal_listener_5epoch.params"):
    
    # Initializing featurizers
    print("Initializing featurizers")
    caption_phi = caption_featurizers.CaptionFeaturizer(tokenizer=caption_featurizers.EndingTokenizer) # using endings tokenizer to separately
    color_phi = ColorFeaturizer(color_phi_fourier, "rgb", normalized=True)
    feature_handler = FeatureHandler(train_data, dev_data, caption_phi, color_phi) # target function is initialized by default

    print("Obtaining training features") # - have to get train features to get vocab size (EVEN IF YOU'RE RUNNING PRETRAINED MODEL)
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
        print("Training model:")
        model.fit(train_features, train_targets)
        model.save_model("model/literal_listener_5epoch-sample.params")
    else:
        print("Loading pretrained model")
        model.load_model(model_file)

    # convert the model output to a score for that particular round
    print("Evaluating model")
    output_to_score = lambda model_outputs, targets: np.exp(model_outputs[np.arange(len(model_outputs)), targets]) # get the model's predicted probablity at each target index and use that as the score
    evaluate_model(dev_data, feature_handler, model, output_to_score, score_model)

# 2. Literal Listener (trained with listener selections as target) 
#    Everything is the same other than the target function
def literal_listener_experiment(train=False, model_file="model/listener_click_predictor_5epoch.params"):
    
    # Initializing featurizers
    print("Initializing featurizers")
    caption_phi = caption_featurizers.CaptionFeaturizer(tokenizer=caption_featurizers.EndingTokenizer) # using endings tokenizer to separately
    color_phi = ColorFeaturizer(color_phi_fourier, "rgb", normalized=True)

    # Now we have a different target function, because we want to predict what the listener selected
    def listener_click_target(data_entry, color_perm):
        # color_perm because colors are randomized
        return np.where(color_perm==data_entry.click_idx)[0]

    feature_handler = FeatureHandler(train_data, dev_data, caption_phi, color_phi, target_fn=listener_click_target) 

    print("Obtaining training features") # - have to get train features to get vocab size (EVEN IF YOU'RE RUNNING PRETRAINED MODEL)
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
        print("Training model:")
        model.fit(train_features, train_targets)
        model.save_model("model/listener_click_predictor_5epoch-sample.params")
    else:
        print("Loading pretrained model")
        model.load_model(model_file)

    # convert the model output to a score for that particular round
    print("Evaluating model")
    output_to_score = lambda model_outputs, targets: np.exp(model_outputs[np.arange(len(model_outputs)), targets]) # get the model's predicted probablity at each target index and use that as the score
    evaluate_model(dev_data, feature_handler, model, output_to_score, score_model)
