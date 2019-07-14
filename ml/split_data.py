import pandas as pd
from tqdm import tqdm


# Split train.csv into train2.csv and test2.csv. Files have similar percentage of
# toxic and non toxic messages. Train2.csv will be used for training and test2.csv
# will only be used for testing

train_df = pd.read_csv("./train.csv")

toxic = 0
notToxic = 0

# precalculated values. hardcoded for efficiency
toxic = 80810
notToxic = 1225312

indexToxic = 0
indexNotToxic = 0

trainData = []
testData = []

for index, row in tqdm(train_df.iterrows()):

    data = {}
    data["qid"] = row.qid
    data["question_text"] = row.question_text
    data["target"] = row.target
    
    if row.target == 0:
        if indexNotToxic < int(notToxic / 2):
            trainData.append(data)
        else:
            testData.append(data)
        indexNotToxic = indexNotToxic + 1

    if row.target == 1:
        if indexToxic < int(toxic / 2):
            trainData.append(data)
        else:
            testData.append(data)
        indexToxic = indexToxic + 1

trainDf = pd.DataFrame(trainData)
testDf = pd.DataFrame(testData)

trainDf.to_csv("train2.csv", index=False)
testDf.to_csv("test2.csv", index=False)