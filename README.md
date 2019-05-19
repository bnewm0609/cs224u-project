# CS224U Project

So, this is the code that's been written so far. Some of it is commented and hopefully the rest will be soon. Basically right now what we have is a simple bunch of scripts that store the data (`monroe_data.py`) for visualization and piping into the model feature functions and the model itself. Right now the file parses a csv file into a pandas df and then adds some features on top of it for visualization and easy color access. Next, the feature functions for the captions are defined in `caption_featurizers.py` and the feature functions for the colors are in `color_featurizers.py`. Currently I think there is one of each. The caption feature function also includes the tokenizer and an "indexer" that maps vocab items to indices. Next is `models.py` which right now just consists of an implementation of the literal listener model from Will Monroe's paper. Finally, `experiment.py` brings everything together. Things that might be useful later on would be to make this work with sklearn stuff, but no consideration has been given to that at all (and probably at the point we should just build on top of Chris' code.)

For examples of how I've been using this code, you can look at the `if __name__ == '__main__'` blocks in `experiment.py` and `monroe_data.py` as well as the python notebooks in the `notebooks` directory (though those are definitely a mess)

The data itself is stored in the `data` directory. In the `csv` subdirectory lives the train/dev/test split data that Monroe used as well as the entire dataset in a single csv called `filteredCorups.csv`. In the `entries` subdirectory are pickle files that each store a list of `MonroeDataEntries` (from `monroe_data.py`) for easy loading into a `MonroeData` object that is passed to a FeatureHandler from `experiment.py`. (It might take 1-2 minutes to recreate them). An example of how to load them is in the `monroe_data.py` block.

The only pretrained model has its weights stored in the `models` directory.

If you have any questions about how to use any of this stuff please feel free to ask, and if you see a place for improvement, please make changes.

Also most of this code is based off the code from: <https://github.com/futurulus/colors-in-context>
