"""SARI, vendored from the HuggingFace `datasets` metric that metrics.py used.

`datasets.load_metric("sari")` no longer exists (and compute nodes cannot
download metric scripts), so this is a local copy of that implementation:
https://github.com/huggingface/datasets/blob/2.14.0/metrics/sari/sari.py
which is itself adapted from the tensor2tensor / Xu et al. (2016) SARI, with
the keep-score recall fix. Scores are on a 0-100 scale, like the original.
"""

from collections import Counter

import sacremoses
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from sacrebleu.tokenizers.tokenizer_intl import TokenizerV14International


def SARIngram(sgrams, cgrams, rgramslist, numref):
    rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
    rgramcounter = Counter(rgramsall)

    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter()
    for sgram, scount in sgramcounter.items():
        sgramcounter_rep[sgram] = scount * numref

    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter()
    for cgram, ccount in cgramcounter.items():
        cgramcounter_rep[cgram] = ccount * numref

    # KEEP
    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter

    keeptmpscore1 = 0
    keeptmpscore2 = 0
    for keepgram in keepgramcounter_rep:
        keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
        # Fix an alleged bug [2] in the keep score computation.
        keeptmpscore2 += keepgramcountergood_rep[keepgram]
    # Define 0/0=1 instead of 0 to give higher scores for predictions that match
    # a target exactly.
    keepscore_precision = 1
    keepscore_recall = 1
    if len(keepgramcounter_rep) > 0:
        keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)
    if len(keepgramcounterall_rep) > 0:
        # Fix an alleged bug [2] in the keep score computation.
        keepscore_recall = keeptmpscore2 / sum(keepgramcounterall_rep.values())
    keepscore = 0
    if keepscore_precision > 0 or keepscore_recall > 0:
        keepscore = 2 * keepscore_precision * keepscore_recall / (keepscore_precision + keepscore_recall)

    # DELETION
    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - rgramcounter
    deltmpscore1 = 0
    for delgram in delgramcounter_rep:
        deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
    # SARI uses deletion precision only; the deletion-recall sum is never used.
    # Define 0/0=1 instead of 0 to give higher scores for predictions that match
    # a target exactly.
    delscore_precision = 1
    if len(delgramcounter_rep) > 0:
        delscore_precision = deltmpscore1 / len(delgramcounter_rep)

    # ADDITION
    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = set(addgramcounter) & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)

    addtmpscore = 0
    for addgram in addgramcounter:
        if addgram in addgramcountergood:
            addtmpscore += 1

    # Define 0/0=1 instead of 0 to give higher scores for predictions that match
    # a target exactly.
    addscore_precision = 1
    addscore_recall = 1
    if len(addgramcounter) > 0:
        addscore_precision = addtmpscore / len(addgramcounter)
    if len(addgramcounterall) > 0:
        addscore_recall = addtmpscore / len(addgramcounterall)
    addscore = 0
    if addscore_precision > 0 or addscore_recall > 0:
        addscore = 2 * addscore_precision * addscore_recall / (addscore_precision + addscore_recall)

    return (keepscore, delscore_precision, addscore)


def _ngrams(tokens, n):
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def SARIsent(ssent, csent, rsents):
    numref = len(rsents)

    s1grams = ssent.split(" ")
    c1grams = csent.split(" ")
    r1gramslist = [rsent.split(" ") for rsent in rsents]

    keeps, dels, adds = [], [], []
    for n in (1, 2, 3, 4):
        sgrams = _ngrams(s1grams, n)
        cgrams = _ngrams(c1grams, n)
        rgramslist = [_ngrams(r1grams, n) for r1grams in r1gramslist]
        keep, delete, add = SARIngram(sgrams, cgrams, rgramslist, numref)
        keeps.append(keep)
        dels.append(delete)
        adds.append(add)

    avgkeepscore = sum(keeps) / 4
    avgdelscore = sum(dels) / 4
    avgaddscore = sum(adds) / 4
    finalscore = (avgkeepscore + avgdelscore + avgaddscore) / 3
    return finalscore


_TOKENIZERS = {"13a": Tokenizer13a, "intl": TokenizerV14International}


def normalize(sentence, lowercase: bool = True, tokenizer: str = "13a", return_str: bool = True):
    # Normalization is required for the ASSET dataset (one of the primary
    # datasets in sentence simplification) to allow using space
    # to split the sentence. Even though Wiki-Auto and TURK datasets,
    # do not require normalization, we do it for consistency.
    if lowercase:
        sentence = sentence.lower()

    if tokenizer in _TOKENIZERS:
        normalized_sent = _TOKENIZERS[tokenizer]()(sentence)
    elif tokenizer == "moses":
        normalized_sent = sacremoses.MosesTokenizer().tokenize(sentence, return_str=True, escape=False)
    elif tokenizer == "penn":
        normalized_sent = sacremoses.MosesTokenizer().penn_tokenize(sentence, return_str=True)
    else:
        normalized_sent = sentence

    if not return_str:
        normalized_sent = normalized_sent.split()

    return normalized_sent


def compute_sari(sources, predictions, references):
    """Corpus SARI in [0, 100]; `references` is a list of reference lists."""
    if not (len(sources) == len(predictions) == len(references)):
        raise ValueError("Sources length must match predictions and references lengths.")
    sari_score = 0
    for src, pred, refs in zip(sources, predictions, references):
        sari_score += SARIsent(normalize(src), normalize(pred), [normalize(sent) for sent in refs])
    sari_score = sari_score / len(predictions)
    return {"sari": 100 * sari_score}
