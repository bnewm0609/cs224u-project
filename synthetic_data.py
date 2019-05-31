# Used for creating synthetic data by switching what the target color is but keeping the same utterance
# This if fine for evaluation, but not for training. This is because once we switch the target color
# the speaker is no longer cooperative. Our metric models should be able to pick up on this, but they shouldn't
# be trained on non-cooperative data because this is not actually how people communicate

from monroe_data import MonroeData, MonroeDataEntry, Color # last two for reading pkl file
import copy
import random
import pandas as pd
import pickle as pkl

def create_synth_data(monroe_df, fake_entries, correct_entries, num_divisions = 5, filename = "synthdata"):
    """
    Given lists of fake and real MonroeDataEntries divided up by condition, create fake speaker
    rounds (and group them into games). Save the created games in a new CSV with the same data
    as the passed CSV except for the values related to how the listener did

    Params
    monroe_df     - a MonoroeData object that contains the dataframe that the synthetic
                    data is based on.
    fake_entries  - a list containing three lists. Each list contains synthetic 
                    MonroeDataEntries for one of the conditions (far/close/split)
    correct_entries - similar to above. A list containing three lists. Each list
                      contains actual MonroeDataEntries associated with rounds 
                      that the listener got correct
    num_divisions -   Denominator for fractions of fake data to include (i.e. if
                      num_divisions is 5, there will be 6 'levels' of speakers: those who
                      are wrong 0/5ths of the time, 1/5 of the time ..., 4/5ths of the
                      time and 5/5ths of the time.
    filename      -   the core part of the file where entries and csv are saved. 
                      (Don't need to include any path information)
    """

    synth_data = []
    condition_counter_fake = 0
    condition_counter_correct = 0
    min_condition_samples = min(min([len(samples) for samples in fake_entries]), min([len(samples) for samples in correct_entries])) # 4173 - train # dev - 4320 # we have the fewest number of correct close samples
    num_games = 3*min_condition_samples//((num_divisions + 1) * 25)

    # let's divide into fifths
    for i in range(num_divisions + 1): # 0...num_divisions inclusive
        for game_id in range(num_games): # 150 size of each group required over whole game over dataset, 3*4320
            synth_game = []
            for j in range(50):
                if j < (50/num_divisions) * i: # add fake stuff
                    synth_game.append(fake_entries[condition_counter_fake % 3][condition_counter_fake // 3])
                    condition_counter_fake += 1
                else: # add real stuff
                    synth_game.append(correct_entries[condition_counter_correct % 3][condition_counter_correct // 3])
                    condition_counter_correct += 1
            random.shuffle(synth_game)
            synth_data.extend(synth_game)

    # construct a dataframe and populate it (this takes a long time)
    # the color information won't be right, but the rest (i.e. click time/content) should be usable
    synth_data_df = pd.DataFrame(index=list(range(len(synth_data))), columns=monroe_df.data.columns)

    for round_counter, sde in enumerate(synth_data):
        idx = sde.index
        game_id = "synth-%d"%(round_counter // 50)
        synth_data_df.loc[round_counter] = monroe_df.data.loc[idx]
        synth_data_df.loc[round_counter, "gameid"] = game_id
        synth_data_df.loc[round_counter, "roundNum"] = round_counter % 50 + 1 # they one index rounds so we do the same...
        synth_data_df.loc[round_counter, "outcome"] = sde.outcome
        synth_data_df.loc[round_counter, "numOutcome"] = 1 if sde.outcome else 0

    with open("data/entries/%s.pkl"%filename, "wb") as pkl_file:
        pkl.dump(synth_data, pkl_file)
    synth_data_df.to_csv("data/csv/%s.csv"%filename, header=True, index=False)

    # load files just saved as MonroeData object for traditional access
    monroe_synth_data = MonroeData("data/csv/%s.csv"%filename, "data/entries/%s.pkl"%filename)
    return monroe_synth_data


def generate_synth_entries(monroe_df):
    """
    First the correct entries are extracted. Then synthetic entries are created by swapping the 
    target color but keeping the caption. In the far condition this corresponds to the speaker
    being not cooperative, and in the close condition this corresponds to the speaker not 
    necessarily being as clear as possible (but potentially not cooperative as well).
    Next, the correct entries are

    Params
    monroe_df - A MonroeData object that the synthetic MonroeDataEntries will be generated from.

    Returns
    fake_entries - a 3 element list where the first element is a list of the synthetic "far" entries,
                   the second is the synthetic "close" entries and the third is the split ones.
    correct_entries - same as fake_entries, but all of the entries come for the actual listener actions.
    """
    # get correct entries
    correct_entries = list(filter(lambda de: de.target_idx == de.click_idx, monroe_df.entries))

    fake_far_entries   = []
    fake_close_entries = []
    fake_split_entries = []

    for de in correct_entries:
        # if close or far, basically symmetric so add both (far fakes will be misleading, close fakes will be ambiguous)
        if de.condition == "far" or de.condition == "close":
            for i in range(1,3):
                de_cp = copy.deepcopy(de)
                
                # instead of switching the target, switch the selected color and the order of the colors. This is so
                # the fake target is still at index 0.
                de_cp.click_idx = i
                de_cp.colors[0], de_cp.colors[i] = de_cp.colors[i], de_cp.colors[0] # swap target and clicked colors

                de_cp.outcome = False
                if de.condition == "far":
                    fake_entries = fake_far_entries
                else: # de.condition == "close"
                    fake_entries = fake_close_entries

                # spread out the same caption/context pair (we're probably only going to ever use the first half)
                if de.index % 2 == 0:
                    fake_entries.append(de_cp) 
                else:
                    fake_entries.insert(0, de_cp)

        # we have to be a bit more careful, because the fake target in the split condition really should
        # be the one that is closest to the target
        elif de.condition == "split":
            # get index to lookup in dataframe
            idx = de.index
            de_cp = copy.deepcopy(de)
            de_cp.outcome = False
            click_idx = 2 # assume the close one is the 2nd distractor
            if monroe_df.data["targetD1Diff"][idx] < monroe_df.data["targetD2Diff"][idx]:
                # D1 is the close distractor (and new target) - corresponds to color at index 1
                click_idx = 1
            de_cp.click_idx = click_idx
            de_cp.colors[0], de_cp.colors[click_idx] = de_cp.colors[click_idx], de_cp.colors[0]
            fake_split_entries.append(de_cp)

    # we also need the correct versions of these tasks
    correct_entries_cond = [[], [], []]
    cond_map = {"far": 0, "close": 1, "split": 2}
    for de in correct_entries:
        correct_entries_cond[cond_map[de.condition]].append(de)
    return [fake_far_entries, fake_close_entries, fake_split_entries], correct_entries_cond


if __name__ == "__main__":
    dev_data = MonroeData("data/csv/dev_corpus_monroe.csv", "data/entries/dev_entries_monroe.pkl")
    print("Obtaining Synthetic Entries")
    dev_data_synth_fake_entries, dev_data_synth_correct_entries = generate_synth_entries(dev_data)
    print("Packaging Synthetic Entries into CSV/pkl files")
    train_data_synth_10 = create_synth_data(dev_data, dev_data_synth_fake_entries, dev_data_synth_correct_entries, num_divisions=10, filename = "dev_corpus_synth_10fold")
