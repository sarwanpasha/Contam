"""Geneformer rank-value tokenization of an AnnData object.

Genecorpus-30M reference cells are stored as rank-ordered token lists produced by
Geneformer's tokenizer, which ranks genes by
    (raw_count / gene_median)   [nonzero only, truncated to 2048]
over a fixed 25,424-Ensembl-ID vocabulary. Benchmark cells are encoded the same way so
that reference and benchmark cells share one input format.
The per-cell scale factors (n_counts, target_sum) are constants within a cell
and therefore do not affect the ranking, so we omit them.
"""
from __future__ import annotations
import pickle
import numpy as np
import scipy.sparse as sp

GF_MAX_LEN = 2048


class GeneformerVocab:
    def __init__(self, median_pkl, token_pkl, name_id_pkl=None):
        self.gene_median = pickle.load(open(median_pkl, 'rb'))
        self.gene_token = pickle.load(open(token_pkl, 'rb'))
        self.name_to_id = pickle.load(open(name_id_pkl, 'rb')) if name_id_pkl else {}
        self.ensembl_ids = [g for g in self.gene_median]

    def resolve(self, var, id_col=None):
        """Map an AnnData .var to Ensembl IDs; returns (ids, how) per gene."""
        ids, how = [], []
        for i in range(len(var)):
            raw_id = str(var[id_col].iloc[i]) if id_col else None
            sym = str(var.index[i])
            cand = None; tag = 'unmapped'
            if raw_id:
                base = raw_id.split('.')[0]
                if base in self.gene_median:
                    cand, tag = base, 'ensembl'
            if cand is None and sym in self.name_to_id:
                m = self.name_to_id[sym]
                if m in self.gene_median:
                    cand, tag = m, 'symbol'
            if cand is None:
                base = sym.split('.')[0]
                if base.startswith('ENSG') and base in self.gene_median:
                    cand, tag = base, 'ensembl_index'
            ids.append(cand); how.append(tag)
        return np.array(ids, dtype=object), np.array(how, dtype=object)


def tokenize(adata, vocab, id_col=None, max_len=GF_MAX_LEN):
    """Return (list of token-index arrays, mapping report dict)."""
    ids, how = vocab.resolve(adata.var, id_col=id_col)
    keep = np.where(ids != None)[0]  # noqa: E711
    # collapse duplicate Ensembl IDs: keep first occurrence (Geneformer behaviour)
    seen, uniq = set(), []
    for j in keep:
        if ids[j] not in seen:
            seen.add(ids[j]); uniq.append(j)
    uniq = np.array(uniq, dtype=int)
    med = np.array([vocab.gene_median[ids[j]] for j in uniq], dtype=np.float64)
    tok = np.array([vocab.gene_token[ids[j]] for j in uniq], dtype=np.int64)
    X = adata.X[:, uniq]
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsr()
    out = []
    for r in range(X.shape[0]):
        s, e = X.indptr[r], X.indptr[r + 1]
        cols, vals = X.indices[s:e], X.data[s:e].astype(np.float64)
        nz = vals > 0
        cols, vals = cols[nz], vals[nz]
        if cols.size == 0:
            out.append(np.empty(0, dtype=np.int64)); continue
        score = vals / med[cols]
        order = np.argsort(-score, kind='stable')[:max_len]
        out.append(tok[cols[order]])
    rep = {
        'n_var': len(adata.var),
        'n_mapped': int(len(keep)),
        'n_unique_tokens': int(len(uniq)),
        'vocab_size': len(vocab.gene_median),
        'frac_var_mapped': float(len(keep)) / len(adata.var),
        'frac_vocab_covered': float(len(uniq)) / len(vocab.gene_median),
        'by_method': {k: int((how == k).sum()) for k in set(how.tolist())},
    }
    return out, rep
