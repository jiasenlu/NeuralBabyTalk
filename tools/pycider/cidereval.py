# coding: utf-8

# In[1]:

# demo script for running CIDEr
import json
from pydataformat.loadData import LoadData
from pyciderevalcap.eval import CIDErEvalCap as ciderEval

# load the configuration file
config = json.loads(open('params.json', 'r').read())

# Print the parameters
print("""Running CIDEr with the following settings
*****************************
Reference File:{refName}
Candidate File:{candName}
Result File:{resultFile}
IDF:{idf}
*****************************""".format(**config))

pathToData = config['pathToData']
refName = config['refName']
candName = config['candName']
resultFile = config['resultFile']
df_mode = config['idf']

# In[2]:

# load reference and candidate sentences
loadDat = LoadData(pathToData)
gts, res = loadDat.readJson(refName, candName)


# In[3]:

# calculate cider scores
scorer = ciderEval(gts, res, df_mode)
# scores: dict of list with key = metric and value = score given to each
# candidate
scores = scorer.evaluate()


# In[7]:

# scores['CIDEr'] contains CIDEr scores in a list for each candidate
# scores['CIDErD'] contains CIDEr-D scores in a list for each candidate

with open(resultFile, 'w') as outfile:
    json.dump(scores, outfile)
