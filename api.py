import spacy
import numpy as np
import textdata
import json
import ast
import math


# load stuff
nlp = spacy.load('en')

with open('stop_words.txt', 'r') as f:
    stop_words = ast.literal_eval(f.read())

with  open('freqs.json', 'r') as f:
    freqs_list = json.load(f)

freq_dict = {}
for x in freqs_list:
    w = x[0].lower().split('|')[0]
    if w in freq_dict:
        freq_dict[w] += x[1]
    else:
        freq_dict[w] = x[1]

freq_sum = sum(list(freq_dict.values()))


associations = []
#associations.append(('born', 'date of birth'))
## add more associations here

def global_frequency(x):
    x = x.lower()
    if x[-2:] == '\'s':
        x = x[:-2]
    if x in freq_dict:
        return float(freq_dict[x]) / freq_sum
    return 0


def local_frequency(x, doc):
    if x[-2:] == '\'s':
        x = x[:-2]
    y = doc.count(x) * 5. / len(doc)
    if y > 1:
        y = 1.
    return y

def tokenize(x):
    # TODO
    x = unicode(x)
    return [w.text for w in nlp(x)]


def frequency(x, doc):
    return global_frequency(x) * local_frequency(x, doc)

oov = {}


def in_vocab(x):
    x = unicode(x)
    if x not in nlp.vocab:
        return False
    emb = nlp(x).vector
    if not any(emb):
        return False
    return True

def _char_embedding(x):
    vec = np.zeros(36)
    if not in_vocab(x):
        return vec
    idxs = list(range(48, 57)) + list(range(97, 123))
    for c in x:
        for i, idx in enumerate(idxs):
            if ord(c) == idx:
                vec[i] = 1
    return vec

def embedding(x):
    x = x.lower()
    if x in oov:
        return oov[x]
    ux = unicode(x)
    if not in_vocab(ux):
        #v = np.random.uniform(-1, 1, embedding_dim)
        v = np.zeros(embedding_dim)
        oov[x] = v
        return v
    x = nlp(ux)
    v = x.vector
    if not np.any(v):
        return v
    nv = x.vector_norm
    return v / nv

def _embedding(x):
    return np.concatenate([_embedding(x), _char_embedding(x)])
def stop_filter(x):
    return [w for w in x if w.lower() not in stop_words]

def doc2vec(x):
    tokens = tokenize(x)
    tokens = stop_filter(tokens)
    tokens = list(set(tokens))
    embs = [embedding(x) for x in tokens]
    oov = [t if not in_vocab(t) else None for t in tokens]
    return np.array(embs), oov

def sim(x, y):
    x, x_oov = x
    y, y_oov = y
    oov_M = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            xo = x_oov[i]
            yo = y_oov[j]
            if xo is not None and xo == yo:
                oov_M[i, j] = 1
    M = np.dot(x, y.T)
    M += oov_M * (np.max(M) + 1)
    max1 = np.max(np.sum(M, axis=0))
    max2 = np.max(np.sum(M, axis=1))

    s = (max1 + max2) / (len(x) + len(y))
    return s


embedding_dim = embedding('the').shape[-1]
stop_word_embs = [embedding(x) for x in stop_words]
stop_words_mean = np.mean(stop_word_embs)
del stop_word_embs

def _information_density(x):
    try:
        float(x)
        return 1
    except Exception:
        if not in_vocab(x):
            return 0
        else:
            emb = embedding(x)
            info = np.sum((emb - stop_words_mean) ** 2) ** 0.5
            return info

def cos_dist(x, y):
    return (((x - y) ** 2).sum()) ** 0.5

def preprocess(x):
    # ascii chars only
    y = [c for c in x if ord(c) in range(128)]
    return ''.join(y)

def find(x, doc):
    # Since some characters are removed during
    # different stages of preprocessing, it is
    # hard to keep track of the index map of 
    # characters. (We have to return the indices
    # of start and characters of answers, not the
    # answers themselves.) So instead of maintaining
    # an index map, we do a soft search for the final
    # answer in the original document.
    # Slightly inefficient (O(n^4)).
    # e.g: find('abcd', 'qwerty,ab,cd,xyz') => (8, 13)
    start = 0
    end = 0
    m = len(x)
    n = len(doc)
    best_dist = n
    for i in range(n):
        for j in range(i + int(m / 2), min(n, i + m * 2)):
            b = doc[i : j]
            a = x
            if len(a) > len(b):
                a, b = b, a
            d1 = list(range(len(a) + 1))
            for ib, cb in enumerate(b):
                d2 = [ib + 1]
                for ia, ca in enumerate(a):
                    if ca == cb:
                        d2.append(d1[ia])
                    else:
                        d2.append(1 + min((d1[ia], d1[ia + 1], d2[-1])))
                d1 = d2
            dist = d1[-1]
            if  dist < best_dist:
                best_dist = dist
                start = i
                end = j
    return start, end


def comp_word(x, y):
    if x == y:
        return 1
    if x in y or y in x:
        return 0.9
    x_t = tokenize(x)
    y_t = tokenize(y)
    c = [t for t in x_t if t in y_t]
    if len(c) > 0:
        return 2. * len(x) / (len(x) + len(y))
    return sim(doc2vec(x), doc2vec(y))


def winner_takes_all(X):
    n = len(X)
    n_half = int(math.ceil(float(n) / 2))
    M = np.zeros((n, n))
    for i in range(n_half):
        cols = list(range(n))
        cols.remove(i)
        for j in cols:
            M[i, j] = comp_word(X[i], X[j])
    for i in range(n_half, n):
        M[j, i] = M[i, j]
    scores = np.sum(M, 1)
    return np.argmax(scores)

def has_information(x):
    abc = list(range(97, 123))
    digs = list(range(48, 57))
    allowed = abc + digs
    x = [c for c in x if ord(c) in allowed]
    return len(x) > 0

def information(x):
    if ' ' in x:
        return sum([information(w) for w in x.split(' ')])
    if not has_information(x):
        return 0
    else:
        return _information_density(x)

def ask(query, docs):
    query = query.lower()
    x = doc2vec(query)
    query_tokens = tokenize(query)
    processed_docs = [preprocess(doc) for doc in docs]
    per_doc_answers = []
    per_doc_scores = []
    for idx, doc in enumerate(processed_docs):
        doc_lower = doc.lower()
        lines = textdata.lines(doc)
        if len(lines) == 0:
            continue
        per_line_scores = [sim(doc2vec(line), x) for line in lines]
        max_idx = np.argmax(per_line_scores)
        best_line_score = per_line_scores[max_idx]
        best_line = lines[max_idx]
        best_line_idxs = find(best_line, doc)
        line_original = doc[best_line_idxs[0] : best_line_idxs[1]]
        tokens = tokenize(best_line)
        tokens = stop_filter(tokens)
        tokens = [t.lower() for t in tokens]
        tokens = [t for t in tokens if t not in query_tokens]
        if len(tokens) == 0:
            continue
        embs = [embedding(t) for t in tokens]
        dists = np.array([cos_dist(e, np.mean(x[0], 0)) for e in embs])
        freqs = np.array([frequency(t, doc_lower) for t in tokens])
        scores = dists * freqs
        min_score = min(scores)
        attended = []
        attended_idxs = []
        for s, t in zip(scores, tokens):
            if s == min_score:
                attended.append(t)
                att_idxs = find(t, line_original)
                att_idxs = (best_line_idxs[0] + att_idxs[0], best_line_idxs[0] + att_idxs[1])
                attended_idxs.append(att_idxs)
        attended_info = information(' '.join(attended))
        S = 2 * attended_info + best_line_score
        answer = {}
        answer['doc'] = doc
        answer['doc_id'] = idx
        answer['snippet_confidence'] = best_line_score
        answer['snippet'] = best_line
        answer['snippet_idxs'] = best_line_idxs
        answer['highlight'] = attended
        answer['highlight_idxs'] = attended_idxs
        answer['highlight_confidence'] = attended_info
        answer['net_confidence'] = S
        per_doc_scores.append(S)
        per_doc_answers.append(answer)
    best_answer = per_doc_answers[np.argmax(per_doc_scores)]
    return best_answer