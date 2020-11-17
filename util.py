import pandas as pd

def extract_all_persona(dataset):
    persona_set = set()
    for d in dataset:
        for _, p in d['pa'].items():
            if not pd.isnull(p) and len(p) > 0:
                persona_set.add(p)

        for _, p in d['pb'].items():
            if not pd.isnull(p) and len(p) > 0:
                persona_set.add(p)

    return list(persona_set)

def extract_all_samples(dataset):
    samples = []
    utterances = []
    personas = []
    for d in dataset:
        ua = [u['A'][0] for u in d['dialogue']]
        ub = [u['B'][0] for u in d['dialogue']]
        pa = [p for _, p in d['pa'].items() if not pd.isnull(p) and len(p) > 0]
        pb = [p for _, p in d['pb'].items() if not pd.isnull(p) and len(p) > 0]
        samples.append((ua, pa))
        samples.append((ub, pb))

        utterances += ua + ub
        personas += pa + pb


    return samples, utterances, personas

def count_utterance(dataset):
    u_count = 0
    p_count = 0
    for d in dataset:
        u_count += len(d['dialogue']) * 2
        p_count += len(d['pa'])
        p_count += len(d['pb'])
    return u_count, p_count
