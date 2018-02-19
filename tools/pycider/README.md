Consensus-based Image Description Evaluation (CIDEr Code)
===================

Evaluation code for CIDEr metric. Provides CIDEr as well as
CIDEr-D (CIDEr Defended) which is more robust to gaming effects.

## Important Note ##
CIDEr by default (with idf parameter set to "corpus" mode) computes IDF values using the reference sentences provided. Thus, CIDEr score for a reference dataset with only 1 image will be zero. When evaluating using one (or few) images, set idf to "coco-val-df" instead, which uses IDF from the MSCOCO Vaildation Dataset for reliable results.

## Requirements ##
- java 1.8.0
- python 2.7

For running the ipython notebook file, update your Ipython to [Jupyter](https://jupyter.org/)

## Files ##
./
- cidereval.py (demo script)

./PyDataFormat
- loadData.py (load the json files for references and candidates)

- {$result\_file}.json (file with the CIDEr and CIDEr-D scores)

./pycocoevalcap: The folder where all evaluation codes are stored.
- evals.py: Performs tokenization and runs both the metrics
- tokenizer: Python wrapper of Stanford CoreNLP PTBTokenizer
- cider: CIDEr evaluation codes
- ciderD: CIDEr-D evaluation codes

## Instructions ##
1. Edit the params.json file to contain path to reference and candidate json files, and the result file where the scores are stored<sup>\*</sup>. 
2. Set the "idf" value in params.json to "corpus" if not evaluating on a single image/instance. Set the "idf" value to "coco-val-df" if evaluating on a single image. In this case IDF values from the MSCOCO dataset are used. If using some other corpus, get the document frequencies into a similar format as "coco-val-df", and put them in the data/ folder as a pickle file. Then set mode to the name of the document frequency file (without the '.p' extension).
3. Sample json reference and candidate files are pascal50S.json and pascal_candsB.json
4. CIDEr scores are stored in "scores" variable: scores['CIDEr'] -> CIDEr scores, scores['CIDErD'] -> CIDEr-D scores

<sup>*</sup>Even when evaluating with independent candidate/references (for eg. when using "coco-val-df"), put multiple candidate and reference entries into the same json files. This is much faster than having separate candidate and reference files and calling the evaluation code separately on each candidate/reference file.
## References ##

- PTBTokenizer: We use the [Stanford Tokenizer](http://nlp.stanford.edu/software/tokenizer.shtml) which is included in [Stanford CoreNLP 3.4.1](http://nlp.stanford.edu/software/corenlp.shtml).
- CIDEr: [CIDEr: Consensus-based Image Description Evaluation] (http://arxiv.org/pdf/1411.5726.pdf)

## Developers ##
- Ramakrishna Vedantam (Virgina Tech)

## Acknowledgments ##
- MS COCO Caption Evaluation Team
