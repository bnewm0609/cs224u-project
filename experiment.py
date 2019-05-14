import torch
import torch.nn as nn
import numpy as np
from monroe_data import MonroeData, MonroeDataEntry, Color # last two for reading pkl file
from caption_featurizers import CaptionFeaturizer
from color_featurizers import ColorFeaturizer, color_phi_fourier

import time
import math


class MonroeExperiment:

    def __init__(self, train_data, test_data, featurizers, model, optimizer=torch.optim.Adadelta,
                 criterion=nn.NLLLoss, lr=0.2, num_epochs=30):
        """
        Right now this is kind of ugly because you have to pass literally all of the experiment arguments
        to this constructor. 
        """
        self.train_data = train_data
        self.test_data = test_data
        self.caption_featurizer = featurizers['caption']
        self.color_featurizer = featurizers['color']
        self.model = model

        self.optimizer = optimizer
        self.criterion = criterion

        # misc args:
        self.lr = 0.2
        self.num_epochs = num_epochs

        # for reproducibility, store training pairs
        self.train_pairs = None

        # also make sure the model has been initialized before we do anything
        self.initialized = False

    def train_iter(self, caption_tensor, color_tensor, target, optimizer, criterion):
        """
        Iterates through a single training pair, querying the model, getting a loss and
        updating the parameters. (TODO: addd some kind of batching to this).

        Very much inspired by the torch NMT example/tutorial thingy
        """
        start_states = self.model.init_hidden_and_context()
        input_length = caption_tensor.size(0)
        optimizer.zero_grad()
        loss = 0

        model_output, _, _ = self.model(caption_tensor, start_states, color_tensor)
        model_output = model_output.view(1, -1)

        loss += criterion(model_output, target)
        loss.backward()
        optimizer.step()

        return loss

    def get_pairs(self, data, construct=False):
        """
        Generates "pairs" - tuples of (caption features, color features, target color index)
        for each entry in the specified dataset. Construct is only True if we have not 
        yet constructed the caption indexer, and will signal the caption featurizer to 
        construct the vocab index. It is only going to be true when called in the
        `init_model` function
        """
        # create pairs (caption, colors, target)
        pairs = []
        for entry in data:
            caption_features = self.caption_featurizer.to_tensor(entry.caption, construct=construct)
            color_features = self.color_featurizer.to_tensor(entry.colors)
            color_features, target = self.color_featurizer.shuffle_colors(color_features)

            pairs.append((caption_features, color_features, target))

        return pairs

    # from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    def init_model(self, **model_params):
        """
        Interesting quirk - in most cases, the model takes the vocabulary size
        as an input, but there's no way for the user to know what the vocab
        size is before calling the init_model method. Even though it's kinda ugly,
        what we can do instead is pass all of the model params (named) minus
        the vocab size to this function, and it will create the model. I don't 
        really like this but it works for now
        """
        if self.initialized:
            return

        self.caption_featurizer.construct_featurizer(self.train_data)
        self.train_pairs = self.get_pairs(self.train_data, construct=True)
        model_params['vocab_size'] = self.caption_featurizer.caption_indexer.size
        self.model = self.model(**model_params)
        self.initialized = True

    def train_model(self):
        # training pairs should have already been created by calling init model
        if not self.initialized:
            print("Make sure you initialize the model with the parameters you want")
            return


        optimizer = self.optimizer(lr=self.lr, params=self.model.parameters())
        criterion = self.criterion()

        start_time = time.time()
        store_losses_every = 100
        print_losses_every = 1000
        self.stored_losses = [] # theoretically we store losses so we can plot them later - 
                                # I don't think this part of the code works though. What we
                                # can do instead is take a few thousand or so training examples
                                # out and use them for "evaluation" every 1 epoch or so
        for epoch in range(self.num_epochs):
            print("---EPOCH {}---".format(epoch))
            stored_loss_total = 0
            print_loss_total = 0

            for i, pair in enumerate(self.train_pairs):
                caption, colors, target = pair
                # print(target)

                loss = self.train_iter(caption, colors, target, optimizer, criterion)
                stored_loss_total += loss.item()
                print_loss_total += loss.item()

                if i % print_losses_every == 0:
                    print_loss_avg = print_loss_total / print_losses_every
                    print("{} ({}:{} {:.2f}%) {:.4f}".format(self.asMinutes(time.time() - start_time),
                                                      epoch, i, i/len(self.train_pairs)*100,
                                                      print_loss_avg))
                    print_loss_total = 0

                if i % store_losses_every == 0:
                    stored_loss_avg = stored_loss_total / store_losses_every
                    self.stored_losses.append(stored_loss_avg)
                    stored_loss_total = 0


    def evaluate_iter(self, pair):
        """
        Same as train_iter except don't use an optimizer and gradients or anything
        like that
        """
        with torch.no_grad():
            caption_tensor, color_tensor, target = pair
            start_states = self.model.init_hidden_and_context()
            model_output, _, _ = self.model(caption_tensor, start_states, color_tensor)

            model_output = model_output.view(1, -1)
            if torch.argmax(model_output).item() == target.item():
                return 1, model_output
            else:
                return 0, model_output


    def evaluate_model(self):
        """
        Evaluate model accuracy
        """
        test_pairs = self.get_pairs(self.test_data)
        self.model.eval()

        total_correct = 0
        for pair in test_pairs:
            correct, _ = self.evaluate_iter(pair)
            total_correct += correct

        accuracy = total_correct/len(self.test_data)
        print("Accuracy: {}".format(accuracy))
        return accuracy

    def load_model(self, filename):
        """
        Load model from saved file at filename
        """
        if not self.initialized:
            self.init_model
        self.model.load_state_dict(torch.load(filename))

    def save_model(self, filename):
        """
        Save model to file at filename
        """
        torch.save(self.model.state_dict(), filename)


if __name__ == "__main__":
    # just load a pretrained model and evaluate it on the dev set
    from monroe_data import MonroeData, MonroeDataEntry, Color # last two for reading pkl file
    from caption_featurizers import CaptionFeaturizer
    from color_featurizers import ColorFeaturizer, color_phi_fourier
    from models import CaptionEncoder

    print("Loading training and dev data")
    train_data = MonroeData("data/csv/train_corpus_monroe.csv", "data/entries/train_entries_monroe.pkl")
    dev_data = MonroeData("data/csv/dev_corpus_monroe.csv", "data/entries/dev_entries_monroe.pkl")

    print("Initializing featurizers")
    caption_phi = CaptionFeaturizer()
    color_phi = ColorFeaturizer(color_phi_fourier, "rgb", normalized=True)

    print("Initializing model")
    # model parameters
    embed_dim = 100; hidden_dim = 100; color_dim= 54;# hard coded for example - 54 comes from color fourier phi
    experiment = MonroeExperiment(train_data, dev_data, {'caption':caption_phi, 'color':color_phi}, CaptionEncoder)
    experiment.init_model(embed_dim = embed_dim, hidden_dim=hidden_dim, color_dim=color_dim) # pass model params (minus vocab size)

    # to train: (probably takes about 2 hrs)
    # experiment.train_model()

    print("Loading and evaluating pretrained model")
    experiment.load_model("models/literal_listener.params")
    experiment.evaluate_model()

