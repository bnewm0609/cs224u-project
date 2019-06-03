from scipy import stats # for pearsonr, spearmanr
import numpy as np
import pandas as pd # for handling test data frame
from enum import Enum, auto # really just to try out enums
from skimage import color


# relevant enums for options
class Speaker(Enum):
    BY_GAME_ID = "gameid"
    BY_WORKER_ID = "workerid_uniq"

class Regressor(Enum):
    PEARSON = stats.pearsonr
    SPEARMAN = stats.spearmanr

class Score(Enum):
    SIMPLE = auto()
    COMPOSITE = auto()


def calculate_scores(eval_df, speaker, score=Score.SIMPLE):
    """
    Right now, we just support the "SIMPLE" score which is the mean
    number of correct listener choices. i.e. we are saying the speaker's
    utterance quality is what uniquely determines how the listener does

    (This function is not used right now, but maybe will be later)
    """
    if score == score.SIMPLE:
        return eval_df.groupby(speaker.value).mean()


def score_model(test_data, scores, speaker=Speaker.BY_GAME_ID, regressor=Regressor.PEARSON, score=Score.SIMPLE):
    """
    Assume scores are in the same order as the test data (i.e. 0th row is 0th score) and calculates a regression
    between the scores of the individual games and the scores from the model
    """
    relevant_columns = ["gameid", "roundNum", "numOutcome"]
    if speaker == Speaker.BY_WORKER_ID:
        relevant_columns.append(Speaker.BY_WORKER_ID.value)
    
    if score == Score.COMPOSITE:
        # no support for this yet but probably also need:
        relevant_columns.extend(["contents", "clkTime", "msgTime"])
    
    eval_df = test_data.data[relevant_columns].copy()
    eval_df["model_scores"] = scores # why we need scores to be in same order as rows
    
    
    if score == score.SIMPLE:
        # calculate scores as the mean of the number of successful utterances
        # a speaker has
        true_scores = eval_df.groupby(speaker.value).numOutcome.mean()
    else:
        true_scores = calculate_scores(eval_df, score)
    
    # calculate a model score 
    model_scores = eval_df.groupby(speaker.value).model_scores.mean()
    
    result = regressor(true_scores, model_scores)
    return result
    
def delta_e_dist(color1, color2):
    """color1 and color2 are in rgb space"""
    # do some nice integer conversions
    color1 = np.round(255*color1)
    color2 = np.round(255*color2)
    # convert colors to lab
    color1_lab = color.rgb2lab(np.array([[color1]], dtype=np.uint8)).flatten()
    color2_lab = color.rgb2lab(np.array([[color2]], dtype=np.uint8)).flatten()

    # compute Delta E CIEDE 2000 distance
    return color.deltaE_ciede2000(color1_lab, color2_lab)
