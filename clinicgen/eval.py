#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import json
import os
import re
import time
from unittest.mock import NonCallableMagicMock
import numpy as np
import torch
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from stanza import Pipeline
from tqdm import tqdm
from clinicgen.data.image2text import ToTokenizedTexts
from clinicgen.nli import BERTScorer, SimpleNLI
from clinicgen.utils import data_cuda, RecoverWords
from clinicgen.external.bleu.bleu import Bleu
from clinicgen.external.cider.cider import Cider, CiderScorer
from clinicgen.external.rouge.rouge import Rouge
from clinicgen.external.spice.spice import Spice
from transformers import pipeline

import sys
sys.path.append('/home/otabek.nazarov/Downloads/research/chexpert-labeler')
import loader
from stages import *
from constants import *
from pathlib import Path

QUESTIONS = [
        'Is there pneumonia?',
        'Is there edema?',
        'Is there pneumothorax?',
        'Are there devices?',
        'Is there opacity?',
        'Is there atelectasis?',
        'Is there cardiomegaly?',
        'Is there lung lesion?',
        'Is there consolidation?',
        'Is there fracture?',
    ]

chex_extractor = Extractor(Path('/home/otabek.nazarov/Downloads/research/chexpert-labeler/phrases/mention'), 
                           Path('/home/otabek.nazarov/Downloads/research/chexpert-labeler/phrases/unmention'), 
                           verbose=False)

chex_classifier = Classifier(Path('/home/otabek.nazarov/Downloads/research/chexpert-labeler/patterns/pre_negation_uncertainty.txt'), 
                             Path('/home/otabek.nazarov/Downloads/research/chexpert-labeler/patterns/negation.txt'), 
                             Path('/home/otabek.nazarov/Downloads/research/chexpert-labeler/patterns/post_negation_uncertainty.txt'), 
                             verbose=False)
chex_aggregator = Aggregator(CATEGORIES, verbose=False)

def label(reports):
    """Label the provided report(s)."""
    cur_loader = loader.load.CustomLoader(reports, False)

    # Load reports in place.
    cur_loader.load()
    # Extract observation mentions in place.
    chex_extractor.extract(cur_loader.collection)
    # Classify mentions in place.
    chex_classifier.classify(cur_loader.collection)
    # Aggregate mentions to obtain one set of labels for each report.
    labels = chex_aggregator.aggregate(cur_loader.collection)

    return labels


class EntityMatcher:
    DOC_SEPARATOR = 'DOCSEP'
    ID_SEPARATOR = '__'
    MODE_EXACT = 'exact'
    MODE_NLI = 'nli'
    MODE_NLI_CONTRADICTION = 'nlic'
    MODE_NLI_ENTAILMENT = 'nlie'
    MODE_NLI_ENTAILMENT_HALF = 'nlieh'
    MODE_SEPARATOR = '-'
    NER_BATCH_SIZE = 256
    PENALTY_SIGMA = 6.0

    def __init__(self, sentences, entities, target_types, mode='exact', batch=48, nli=None):
        self.sentences = sentences
        self.entities = entities
        self.target_types = target_types
        self.batch = batch
        self.nli = nli
        self.ner = self.load_ner()
        m = mode.split(self.MODE_SEPARATOR)
        self.mode = m[0]
        if self.mode == self.MODE_NLI_ENTAILMENT_HALF:
            self.mode = self.MODE_NLI_ENTAILMENT
            self.entail_score = 0.5
        else:
            self.entail_score = 1.0
        self.penalty = False
        if len(m) > 1 and m[1] == 'p':
            self.prf = 'p'
        elif len(m) > 1 and m[1] == 'r':
            self.prf = 'r'
        elif len(m) > 1 and m[1] == 'fp':
            self.prf = 'f'
            self.penalty = True
        else:
            self.prf = 'f'

    @classmethod
    def load_entities(cls, path, target_types):
        sentences, entities = {}, {}
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                did, sid = entry['id'].split(cls.ID_SEPARATOR)
                sid = int(sid)
                if did not in sentences:
                    sentences[did] = {}
                sentences[did][sid] = entry['text'].lower()
                if did not in entities:
                    entities[did] = {}
                for entity in entry['nes']:
                    if entity['type'] in target_types:
                        s = entity['text'].lower()
                        if s not in entities[did]:
                            entities[did][s] = [sid]
                        else:
                            entities[did][s].append(sid)
        return sentences, entities

    @classmethod
    def load_ner(cls):
        config = {'tokenize_batch_size': cls.NER_BATCH_SIZE, 'ner_batch_size': cls.NER_BATCH_SIZE}
        return Pipeline(lang='en', package='radiology', processors={'tokenize': 'default', 'ner': 'radiology'},
                        **config)

    def _nli_label(self, prediction):
        best_label, best_prob = 'entailment', 0.0
        for label, prob in prediction.items():
            if prob > best_prob:
                best_label = label
                best_prob = prob
        return best_label, best_prob

    def cuda(self, device=None):
        if self.nli is not None:
            self.nli = self.nli.cuda(device)
        return self

    def score(self, rids, hypos):
        # Named entity recognition
        hypo_sents = {}
        hypos_entities = {}
        texts, buf = [], []
        for hypo in hypos:
            buf.append(hypo)
            if len(buf) >= self.batch:
                text = '\n\n{0}\n\n'.format(self.DOC_SEPARATOR).join(buf)
                texts.append(text)
                buf = []
        if len(buf) > 0:
            text = '\n\n{0}\n\n'.format(self.DOC_SEPARATOR).join(buf)
            texts.append(text)
        i = 0
        for text in texts:
            doc = self.ner(text)
            j = 0
            for sentence in doc.sentences:
                if i not in hypos_entities:
                    hypos_entities[i] = {}
                if i not in hypo_sents:
                    hypo_sents[i] = ''
                if sentence.text == self.DOC_SEPARATOR:
                    i += 1
                    j = 0
                else:
                    if len(hypo_sents[i]) > 0:
                        hypo_sents[i] += '\n'
                    hypo_sents[i] += sentence.text
                    for entity in sentence.ents:
                        if entity.type in self.target_types:
                            buf = []
                            for word in entity.words:
                                buf.append(word.text.lower())
                            s = ' '.join(buf)
                            if s not in hypos_entities:
                                hypos_entities[i][s] = [j]
                            else:
                                hypos_entities[i][s].append(j)
                    j += 1
            i += 1
        hypo_nli, ref_nli = None, None
        if self.mode.startswith(self.MODE_NLI):
            hypo_nli, ref_nli = {}, {}
            texts1, texts2 = [], []
            for i, rid in enumerate(rids):
                try:
                    buf = []
                    rid = rid.split(self.ID_SEPARATOR)[0]
                    for sid in sorted(self.sentences[rid].keys()):
                        buf.append(self.sentences[rid][sid])
                    texts1.append('\n'.join(buf))
                    texts2.append(hypo_sents[i])
                except:
                    continue
            _, _, _, stats = self.nli.sentence_scores_bert_score(texts1, texts2, label='all', prf=self.prf)
            
            for i in range(len(rids)):
                try:
                    rid, rs = rids[i], stats[i]
                    ref_nli[i] = {}
                    for sid, tup in rs['scores'][0].items():
                        pred, _ = self._nli_label(tup[0])
                        ref_nli[i][sid] = pred
                    hypo_nli[i] = {}
                    for sid, tup in rs['scores'][1].items():
                        pred, _ = self._nli_label(tup[0])
                        hypo_nli[i][sid] = pred
                except:
                    continue
        
        # Calculate scores
        scores_e, scores_n = [], []
        for i, rid in enumerate(rids):
            try:
                hypo_entities = hypos_entities[i]
                rid = rid.split(self.ID_SEPARATOR)[0]
                ref_entities = self.entities[rid]
                # precision
                match_e, match_n, total_pr = 0, 0, 0
                if self.prf != 'r':
                    for s in hypo_entities.keys():
                        for sid in hypo_entities[s]:
                            if s in ref_entities:
                                match_e += 1
                                if hypo_nli is None:
                                    match_n += 1.0
                            if hypo_nli is not None:
                                if hypo_nli[i][sid] == 'neutral':
                                    if s in ref_entities:
                                        match_n += 1.0
                                elif hypo_nli[i][sid] == 'entailment':
                                    if s in hypo_entities:
                                        match_n += 1.0
                                    else:
                                        if self.mode == self.MODE_NLI or self.mode == self.MODE_NLI_ENTAILMENT:
                                            match_n += self.entail_score
                                elif hypo_nli[i][sid] == 'contradiction':
                                    if self.mode == self.MODE_NLI_ENTAILMENT:
                                        if s in hypo_entities:
                                            match_n += 1.0
                            total_pr += 1
            except:
                continue
            pr_e = match_e / total_pr if total_pr > 0 else 0.0
            pr_n = match_n / total_pr if total_pr > 0 else 0.0
            # recall
            match_e, match_n, total_rc = 0, 0, 0
            if self.prf != 'p':
                for s in ref_entities.keys():
                    for sid in ref_entities[s]:
                        if s in hypo_entities:
                            match_e += 1
                            if ref_nli is None:
                                match_n += 1.0
                        if ref_nli is not None:
                            if ref_nli[i][sid] == 'neutral':
                                if s in hypo_entities:
                                    match_n += 1.0
                            elif ref_nli[i][sid] == 'entailment':
                                if s in hypo_entities:
                                    match_n += 1.0
                                else:
                                    if self.mode == self.MODE_NLI or self.mode == self.MODE_NLI_ENTAILMENT:
                                        match_n += self.entail_score
                            elif ref_nli[i][sid] == 'contradiction':
                                if self.mode == self.MODE_NLI_ENTAILMENT:
                                    if s in hypo_entities:
                                        match_n += 1.0
                        total_rc += 1
            rc_e = match_e / total_rc if total_rc > 0 else 0.0
            rc_n = match_n / total_rc if total_rc > 0 else 0.0
            # fb1
            if self.prf == 'p':
                score_e, score_n = pr_e, pr_n
            elif self.prf == 'r':
                score_e, score_n = rc_e, rc_n
            else:
                score_e = 2 * pr_e * rc_e / (pr_e + rc_e) if pr_e > 0.0 and rc_e > 0.0 else 0.0
                score_n = 2 * pr_n * rc_n / (pr_n + rc_n) if pr_n > 0.0 and rc_n > 0.0 else 0.0
            if self.penalty:
                penalty = np.e ** (-((total_pr - total_rc) ** 2) / (2 * self.PENALTY_SIGMA ** 2))
                score_e *= penalty
                score_n *= penalty
            scores_e.append(score_e)
            scores_n.append(score_n)
        mean_exact_e = np.mean(scores_e)
        mean_exact_n = np.mean(scores_n)
        return mean_exact_e, scores_e, mean_exact_n, scores_n


class GenEval:
    EVAL_ID = 'id'
    EVAL_REPORT = 'report'
    EVAL_SCORE = 'score'
    EVAL_SCORE_DETAILED = 'score_detailed'
    EVAL_SIZE = 10000
    ID_SEPARATOR = '__'
    NLI_MED = 'mednli'
    NLI_RAD_AUG = 'mednli-rad'
    BERT_SCORE_DEFAULT = 'distilbert-base-uncased'

    LINEBREAK = '__BR__'
    SPLIT_PATTERN = re.compile('[\\s\n]')

    def __init__(self, model, word_indexes, beam_size, bleu=True, rouge=True, cider=True, cider_df=None, spice=False,
                 bert_score=None, bert_score_penalty=False, nli=None, nli_compare=None, nli_label='entailment',
                 nli_neutral_score=(1.0 / 3), nli_prf='f', nli_batch=16, nli_cache=None, entity_match=None,
                 entity_mode='exact', beam_diversity=0.0, nucleus_p=None, nthreads=2, pin_memory=False,
                 sentsplitter='nltk', verbose=False, qa_score=None, chexpert=None):
        self.model = model
        self.recover_words = RecoverWords(word_indexes)
        self.beam_size = beam_size
        self.bleu = bleu
        self.rouge = rouge
        self.cider = cider
        self.cider_df = cider_df
        self.spice = spice
        self.bert_score = bert_score
        self.bert_score_penalty = bert_score_penalty
        self.qa_score = qa_score
        self.chexpert = chexpert
        self.nli = nli
        nli_compare = nli_compare.split(',') if isinstance(nli_compare, str) else nli_compare
        self.nli_batch = nli_batch
        self.nli_compare = nli_compare
        self.nli_label = nli_label
        self.nli_neutral_score = nli_neutral_score
        self.nli_prf = nli_prf
        self.nli_cache = nli_cache
        self.entity_match = entity_match
        self.entity_mode = entity_mode
        self.beam_diversity = beam_diversity
        self.nucleus_p = nucleus_p
        self.nthreads = nthreads
        self.pin_memory = pin_memory
        self.sentsplitter = sentsplitter
        self.verbose = verbose

        self.nli_model = None
        self.bert_score_model = None
        self.entity_matcher = None
        self.device = 'gpu'

    @classmethod
    def _append_eval(cls, rs1, rs2):
        if rs1 is None:
            rs1 = rs2
        else:
            for i in range(len(rs1)):
                if isinstance(rs2[i], float) or isinstance(rs2[i], np.float32):
                    if isinstance(rs1[i], float) or isinstance(rs1[i], np.float32):
                        rs1[i] = [rs1[i], rs2[i]]
                    else:
                        rs1[i] += [rs2[i]]
                else:
                    if isinstance(rs1[i], list):
                        rs1[i] += rs2[i]
                    elif isinstance(rs1[i], np.ndarray):
                        rs1[i] = np.append(rs1[i], rs2[i])
                    else:
                        raise ValueError('Unsupported result {0} in index {1}'.format(type(rs1[i]).__name__, i))
        return rs1

    @classmethod
    def abbreviated_metrics(cls, metrics):
        if not isinstance(metrics, list):
            metrics = [metrics]
        abbrs = []
        for metric in metrics:
            if metric.startswith('BLEU'):
                abbrs.append('BL' + metric[4])
            elif metric == 'ROUGE':
                abbrs.append('RG')
            elif metric == 'CIDEr':
                abbrs.append('CDr')
            elif metric == 'SPICE':
                abbrs.append('SP')
            elif metric.startswith('BERT'):
                abbrs.append('BT-' + metric[-1])
            elif metric == 'NLISentBERTScore':
                abbrs.append('NLI-SB')
            elif metric == 'NLISentBERTScoreT':
                abbrs.append('NLI-SBT')
            elif metric == 'NLISentTFIDF':
                abbrs.append('NLI-TF')
            elif metric == 'NLISentAll':
                abbrs.append('NLI-SA')
            elif metric.startswith('NLI'):
                abbrs.append('NLI-' + metric[3])
            elif metric == 'CheXpertAcc':
                abbrs.append('CXA')
            elif metric == 'EntityMatchExact':
                abbrs.append('EM-E')
            elif metric == 'EntityMatchNLI':
                abbrs.append('EM-N')
            else:
                abbrs.append(metric)
        if len(metrics) == 1:
            abbrs = abbrs[0]
        return abbrs

    @classmethod
    def compute_cider_df(cls, refs):
        scorer = CiderScorer(refs=refs)
        scorer.compute_doc_freq()
        return scorer.document_frequency

    @classmethod
    def compute_tfidf_vectorizer(cls, data_loader):
        refs = []
        for _, _, targ, _, _, _ in data_loader:
            for text in targ:
                refs.append(text)
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 4), min_df=2)
        vectorizer.fit(refs)
        return vectorizer

    @classmethod
    def full_metrics(cls):
        nli_compare = [SimpleNLI.COMPARE_DOC, SimpleNLI.COMPARE_BERT_SCORE, SimpleNLI.COMPARE_BERT_SCORE_FIX_THRESH,
                       SimpleNLI.COMPARE_TFIDF, SimpleNLI.COMPARE_ALL]
        return cls.get_metrics(True, True, True, True, True, True, True, nli_compare, True, True)

    @classmethod
    def get_metrics(cls, bleu, rouge, cider, spice, bert_score, nli, entity_match, nli_compare, qa_score, chexpert):
        m = {}
        if bleu:
            idx = len(m)
            m[idx] = 'BLEU1'
            m[idx + 1] = 'BLEU2'
            m[idx + 2] = 'BLEU3'
            m[idx + 3] = 'BLEU4'
        if rouge:
            m[len(m)] = 'ROUGE'
        if cider:
            m[len(m)] = 'CIDEr'
        if spice:
            m[len(m)] = 'SPICE'
        if bert_score is not None:
            idx = len(m)
            m[idx] = 'BERTScoreP'
            m[idx + 1] = 'BERTScoreR'
            m[idx + 2] = 'BERTScoreF'
        if qa_score is True:
            idx = len(m)
            m[idx] = 'QAScore'
        if chexpert is True:
            idx = len(m)
            m[idx] = 'chexpert'
        if nli is not None:
            if SimpleNLI.COMPARE_DOC in nli_compare:
                idx = len(m)
                m[idx] = 'NLIEntail'
                m[idx + 1] = 'NLINeutral'
                m[idx + 2] = 'NLIContradict'
            if SimpleNLI.COMPARE_BERT_SCORE in nli_compare:
                m[len(m)] = 'NLISentBERTScore'
            if SimpleNLI.COMPARE_BERT_SCORE_FIX_THRESH in nli_compare:
                m[len(m)] = 'NLISentBERTScoreT'
            if SimpleNLI.COMPARE_TFIDF in nli_compare:
                m[len(m)] = 'NLISentTFIDF'
            if SimpleNLI.COMPARE_ALL in nli_compare:
                m[len(m)] = 'NLISentAll'
        if entity_match is not None:
            idx = len(m)
            m[idx] = 'EntityMatchExact'
            m[idx + 1] = 'EntityMatchNLI'
        return m

    @classmethod
    def nli_rewrite(cls, text):
        text = text.replace(" ' ", "'")
        text = text.replace(" n't", "n't")
        text = text.replace(' - ', '-')
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        return text

    @classmethod
    def nli_tfidf(cls, metrics):
        if metrics is None:
            return False
        else:
            if 'NLISentTFIDF' in metrics.split(','):
                return True
            else:
                return False

    def cleanup(self):
        if self.nli_model is not None:
            self.nli_model.stop()
            self.nli_model = None
        if self.bert_score_model is not None:
            self.bert_score_model = None
        if self.entity_matcher is not None:
            self.entity_matcher = None

    def cuda(self, device=None):
        self.device = 'gpu'
        if self.nli_model is not None:
            self.nli_model = self.nli_model.cuda(device)
        if self.bert_score_model is not None:
            self.bert_score_model = self.bert_score_model.cuda(device)
        if self.entity_matcher is not None:
            self.entity_matcher = self.entity_matcher.cuda(device)
        return self

    def eval(self, ids, refs, hypos, tfidf_vectorizer=None, ref_ids=None):
        if ref_ids is None:
            ref_ids = ids
        scores, scores_detailed = [], []
        # BLEU 1-4
        if self.bleu:
            bleu = Bleu(n=4)
            bl, bls = bleu.compute_score(refs, hypos, verbose=-1)
            scores += bl
            scores_detailed += bls
        # ROUGE
        if self.rouge:
            rouge = Rouge()
            rg, rgs = rouge.compute_score(refs, hypos)
            scores.append(rg)
            scores_detailed.append(rgs)
        # CIDEr
        if self.cider:
            cider = Cider(n=4, df=self.cider_df)
            cd, cds = cider.compute_score(refs, hypos)
            scores.append(cd)
            scores_detailed.append(cds)
        # SPICE
        if self.spice:
            spice = Spice()
            sp, sps = spice.compute_score(refs, hypos)
            sps = list(map(lambda v: v['All']['f'], sps))
            scores.append(sp)
            scores_detailed.append(sps)
        # BERTScore
        if self.bert_score_model is not None:
            hypos_l, refs_l = [], []
            for rid in ids:
                hypo = hypos[rid][0]
                hypo = self.nli_rewrite(hypo)
                hypos_l.append(hypo)
                ref = refs[rid][0]
                ref = self.nli_rewrite(ref)
                refs_l.append(ref)
            bp, br, bf = self.bert_score_model.score(hypos_l, refs_l)
            bp, br, bf = bp.numpy(), br.numpy(), bf.numpy()
            scores.append(bp.mean())
            scores.append(br.mean())
            scores.append(bf.mean())
            scores_detailed.append(bp)
            scores_detailed.append(br)
            scores_detailed.append(bf)

        if self.chexpert is True:
            hypos_l, refs_l = [], []
            for rid in ids:
                hypo = hypos[rid][0]
                hypos_l.append(hypo)
                ref = refs[rid][0]
                refs_l.append(ref)
            refs_labels = label(refs_l)
            hypo_labels = label(hypos_l)
            refs_labels = np.nan_to_num(refs_labels)
            hypo_labels = np.nan_to_num(hypo_labels)
            refs_labels = np.where(refs_labels == -1, 1, refs_labels)
            hypo_labels = np.where(hypo_labels == -1, 1, hypo_labels)
            f1_results = [f1_score(cur_ref, cur_hypo, average='binary', zero_division=0) for cur_ref, cur_hypo in zip(refs_labels, hypo_labels)]
            f1_results = np.array(f1_results)
            scores.append(f1_results.mean())
            scores_detailed.append(f1_results)


        # QAScore
        # python train.py --cuda --corpus mimic-cxr --cache-data cache --epochs 32 --batch-size 24 --rl-epoch 1 --rl-metrics QAScore --rl-weights 1.0 --img-model densenet --img-pretrained resources/chexpert_auc14.dict.gz --cider-df mimic-cxr_train-df.bin.gz --lr 5e-6 --lr-step 32 /home/otabek.nazarov/Downloads/ resources/glove_mimic-cxr_train.512.txt.gz out_m2trans_nll-bs-emexact

        # python train.py --cuda --corpus mimic-cxr --cache-data cache --epochs 32 --batch-size 24 --rl-epoch 1 --rl-metrics chexpert --rl-weights 1.0 --img-model densenet --img-pretrained resources/chexpert_auc14.dict.gz --cider-df mimic-cxr_train-df.bin.gz --lr 5e-6 --lr-step 32 /home/otabek.nazarov/Downloads/ resources/glove_mimic-cxr_train.512.txt.gz out_test
        
        if self.qa_model is not None:
            hypos_l, refs_l = [], []
            for rid in ids:
                hypo = hypos[rid][0]
                hypos_l.append(hypo)
                ref = refs[rid][0]
                refs_l.append(ref)
            
            # refs_scores = np.zeros((len(refs_l), len(QUESTIONS)))
            # hypo_scores = np.zeros((len(refs_l), len(QUESTIONS)))
            bert_scores = np.empty((len(refs_l), len(QUESTIONS)))
            bert_scores.fill(np.nan)
            
            for q_idx, cur_question in enumerate(QUESTIONS):    
                # Copy questions for batch forwarding to the model
                question_batch = [cur_question] * len(hypos_l)
                try:
                    # Get results from QA model
                    refs_cur_results = self.qa_model(question=question_batch, context=refs_l)
                    hypo_cur_results = self.qa_model(question=question_batch, context=hypos_l)

                    # Get bert scores for given answers
                    # bert_score_refs = []
                    # bert_score_hypo = []

                    # cider_refs = {}
                    # cider_hypos = {}
                    # for sample_idx, (cur_ref_res, cur_hypo_res, cur_key) in enumerate(zip(refs_cur_results, hypo_cur_results, list(refs.keys()))):
                    #     bert_score_refs.append(cur_ref_res['answer'])
                    #     bert_score_hypo.append(cur_hypo_res['answer'])

                    #     cider_refs[cur_key] = [cur_ref_res['answer']]
                    #     cider_hypos[cur_key] = [cur_hypo_res['answer']]
                    
                    # b_prec, b_recall, b_f1 = self.bert_score_qa_model.score(bert_score_hypo, bert_score_refs)
                    # b_prec, b_recall, b_f1 = b_prec.numpy(), b_recall.numpy(), b_f1.numpy()

                    b_prec, b_recall, b_f1 = self.bert_score_qa_model.score(hypos_l, refs_l)
                    b_prec, b_recall, b_f1 = b_prec.numpy(), b_recall.numpy(), b_f1.numpy()

                    # bleu = Bleu(n=4)
                    # _, bls = bleu.compute_score(cider_refs, cider_hypos, verbose=-1)
                    # bls = np.array(bls)
                    # blue_weighted  = 0.25 * bls[0] + 0.25 * bls[1] + 0.25 * bls[2] + 0.25 * bls[3]

                    # Select scores for loss based on threshold
                    threshold = 0.20
                    for sample_idx, (cur_ref_res, cur_hypo_res) in enumerate(zip(refs_cur_results, hypo_cur_results)):
                        if cur_ref_res['score'] > threshold or cur_hypo_res['score'] > threshold:
                            bert_scores[sample_idx, q_idx] = b_f1[sample_idx]#blue_weighted[sample_idx]#b_f1[sample_idx]
                except:
                    print('Context issue with QA model')
            
            # distances = -1 * np.abs(refs_scores - hypo_scores)
            # # distances = np.abs(refs_scores - hypo_scores) # we want to maximize the score
            # scores.append(distances.sum())
            # # scores_detailed.append((len(QUESTIONS) - distances.sum(axis=1))/len(QUESTIONS))
            # scores_detailed.append(distances.sum(axis=1))
            loss_score = np.nanmean(bert_scores, axis=1)
            loss_score = np.nan_to_num(loss_score)
            scores.append(np.nanmean(bert_scores))
            scores_detailed.append(loss_score)

        # # NLI
        # if self.nli_model is not None:
        #     comps = self.nli_compare
        #     hypos_l, refs_l = [], []
        #     for rid in ids:
        #         hypo = hypos[rid][0]
        #         hypo = self.nli_rewrite(hypo)
        #         hypos_l.append(hypo)
        #         ref = refs[rid][0]
        #         ref = self.nli_rewrite(ref)
        #         refs_l.append(ref)
        #     if SimpleNLI.COMPARE_DOC in comps:
        #         probs, _ = self.nli_model.predict(refs_l, hypos_l)
        #         nli_scores1 = {SimpleNLI.LABEL_ENTAIL: [], SimpleNLI.LABEL_NEUTRAL: [], SimpleNLI.LABEL_CONTRADICT: []}
        #         for prob in probs:
        #             for k, v in prob.items():
        #                 nli_scores1[k].append(v)
        #         probs, _ = self.nli_model.predict(hypos_l, refs_l)
        #         nli_scores2 = {SimpleNLI.LABEL_ENTAIL: [], SimpleNLI.LABEL_NEUTRAL: [], SimpleNLI.LABEL_CONTRADICT: []}
        #         for prob in probs:
        #             for k, v in prob.items():
        #                 nli_scores2[k].append(v)
        #         nli_scores = {SimpleNLI.LABEL_ENTAIL: [], SimpleNLI.LABEL_NEUTRAL: [], SimpleNLI.LABEL_CONTRADICT: []}
        #         for k in [SimpleNLI.LABEL_ENTAIL, SimpleNLI.LABEL_NEUTRAL, SimpleNLI.LABEL_CONTRADICT]:
        #             for v1, v2 in zip(nli_scores1[k], nli_scores2[k]):
        #                 v = 2 * v1 * v2 / (v1 + v2)
        #                 nli_scores[k].append(v)
        #         scores.append(np.mean(nli_scores[SimpleNLI.LABEL_ENTAIL]))
        #         scores.append(np.mean(nli_scores[SimpleNLI.LABEL_NEUTRAL]))
        #         scores.append(np.mean(nli_scores[SimpleNLI.LABEL_CONTRADICT]))
        #         scores_detailed.append(nli_scores[SimpleNLI.LABEL_ENTAIL])
        #         scores_detailed.append(nli_scores[SimpleNLI.LABEL_NEUTRAL])
        #         scores_detailed.append(nli_scores[SimpleNLI.LABEL_CONTRADICT])
        #     if SimpleNLI.COMPARE_BERT_SCORE in comps:
        #         _, _, vs, _ = self.nli_model.sentence_scores(refs_l, hypos_l, SimpleNLI.COMPARE_BERT_SCORE,
        #                                                      label=self.nli_label, prf=self.nli_prf)
        #         scores.append(np.mean(vs))
        #         scores_detailed.append(vs)
        #     if SimpleNLI.COMPARE_BERT_SCORE_FIX_THRESH in comps:
        #         _, _, vs, _ = self.nli_model.sentence_scores(refs_l, hypos_l, SimpleNLI.COMPARE_BERT_SCORE_FIX_THRESH,
        #                                                      label=self.nli_label, prf=self.nli_prf)
        #         scores.append(np.mean(vs))
        #         scores_detailed.append(vs)
        #     if SimpleNLI.COMPARE_TFIDF in comps:
        #         _, _, vs, _ = self.nli_model.sentence_scores(refs_l, hypos_l, SimpleNLI.COMPARE_TFIDF,
        #                                                      tfidf_vectorizer=tfidf_vectorizer, label=self.nli_label)
        #         scores.append(np.mean(vs))
        #         scores_detailed.append(vs)
        #     if SimpleNLI.COMPARE_ALL in comps:
        #         _, _, vs, _ = self.nli_model.sentence_scores(refs_l, hypos_l, SimpleNLI.COMPARE_ALL,
        #                                                      label=self.nli_label, prf=self.nli_prf)
        #         scores.append(np.mean(vs))
        #         scores_detailed.append(vs)
        # Entity Match
        # if self.entity_match is not None:
        #     hypos_l = []
        #     for hid in ids:
        #         hypos_l.append(self.nli_rewrite(hypos[hid][0]))
        #     t = time.time()
        #     mse, sde, msn, sdn = self.entity_matcher.score(ref_ids, hypos_l)
        #     if self.verbose:
        #         print('Entity match {0} pairs: {1}s'.format(len(ids), time.time() - t))
        #     scores.append(mse)
        #     scores.append(msn)
        #     scores_detailed.append(sde)
        #     scores_detailed.append(sdn)
        return scores, scores_detailed

    def eval_batch(self, ids, refs, hypos, tfidf_vectorizer=None, ref_ids=None, batch_size=10000, progress_name=None):
        if progress_name is not None:
            pbar = tqdm(total=len(ids))
            pbar.set_description('{0}'.format(progress_name + '-eval'))
        else:
            pbar = None

        c = 0
        ids_set, refs_set, hypos_set = [], {}, {}
        ref_ids_set = [] if ref_ids is not None else None
        scores, scores_detailed = None, None
        for i, rid in enumerate(ids):
            ids_set.append(rid)
            refs_set[rid] = refs[rid]
            hypos_set[rid] = hypos[rid]
            if ref_ids is not None:
                ref_ids_set.append(ref_ids[i])
            c += 1
            if c >= batch_size:
                s1, s2 = self.eval(ids_set, refs_set, hypos_set, tfidf_vectorizer, ref_ids_set)
                scores = self._append_eval(scores, s1)
                scores_detailed = self._append_eval(scores_detailed, s2)
                if pbar is not None:
                    pbar.update(len(ids_set))
                ids_set, refs_set, hypos_set = [], {}, {}
                ref_ids_set = [] if ref_ids is not None else None
                c = 0
        if c > 0:
            s1, s2 = self.eval(ids_set, refs_set, hypos_set, tfidf_vectorizer, ref_ids_set)
            scores = self._append_eval(scores, s1)
            scores_detailed = self._append_eval(scores_detailed, s2)
            if pbar is not None:
                pbar.update(len(ids_set))
        scores = [np.mean(score) for score in scores]
        return scores, scores_detailed

    def generate_and_eval(self, data_loader, progress_name=None, batch=False):
        # Evaluate generate outputs
        self.model.eval()
        with torch.no_grad():
            if progress_name is not None:
                pbar = tqdm(total=len(data_loader.dataset.samples))
                pbar.set_description('{0}'.format(progress_name + '-gen'))
                eval_interval = int(len(data_loader.dataset.samples) / 10)
            else:
                pbar, eval_interval = None, None
            report_ids, reports, hypos, refs, tqdm_interval = [], [], {}, {}, 0
            for rids, inp, targ, vp in data_loader:
                inp = data_cuda(inp, device=self.device, non_blocking=data_loader.pin_memory)
                meta = (vp,)
                meta = self.model.meta_cuda(meta, device=self.device, non_blocking=data_loader.pin_memory)
                rec_words, _ = self.recover_words if self.verbose else None, None
                encoded_data = self.model.encode(inp, meta)
                if self.nucleus_p is not None:
                    words = []
                    for _ in range(self.beam_size):
                        w, _ = self.model.sample(encoded_data, self.nucleus_p)
                        words.append(w.unsqueeze(dim=1))
                    stops = self.model.dummy_stops(words[0])
                else:
                    stops, words, _ = self.model.decode_beam(encoded_data, self.beam_size, recover_words=rec_words,
                                                             diversity_rate=self.beam_diversity)
                # Output all beams if diversity rate is set
                idxs = list(range(self.beam_size)) if self.beam_diversity > 0.0 or self.nucleus_p is not None else [0]
                for idx in idxs:
                    widxs = words[:, :, idx] if self.nucleus_p is None else words[idx]
                    reps, _ = self.recover_words(stops, widxs)
                    for rid, reference, candidate in zip(rids, targ, reps):
                        # Recovered Samples
                        if self.beam_diversity > 0.0 or self.nucleus_p is not None:
                            rid += '__{0}'.format(idx)
                        report_ids.append(rid)
                        reports.append(candidate.replace('\n', ' ' + self.LINEBREAK + ' '))
                        hypos[rid] = [candidate.replace('\n', ' ')]
                        if data_loader.dataset.multi_instance:
                            reference = reference.split(ToTokenizedTexts.INSTANCE_BREAK)
                        else:
                            reference = [reference]
                        refs[rid] = []
                        for ref in reference:
                            refs[rid].append(ref.replace('\n', ' '))
                tqdm_interval += inp.shape[0]
                if pbar is not None and tqdm_interval >= eval_interval:
                    pbar.update(tqdm_interval)
                    tqdm_interval = 0
            if pbar is not None:
                if tqdm_interval > 0:
                    pbar.update(tqdm_interval)
                pbar.close()
        self.model.train()
        # Calculate IDFs for NLI-TFIDF
        if self.nli is not None and SimpleNLI.COMPARE_TFIDF in self.nli_compare:
            tfidf_vectorizer = self.compute_tfidf_vectorizer(data_loader)
        else:
            tfidf_vectorizer = None
        # Evaluate with metrics
        if batch:
            scores, scores_detailed = self.eval_batch(report_ids, refs, hypos, tfidf_vectorizer,
                                                      batch_size=self.EVAL_SIZE, progress_name=progress_name)
        else:
            scores, scores_detailed = self.eval(report_ids, refs, hypos, tfidf_vectorizer)
        return {self.EVAL_ID: report_ids, self.EVAL_SCORE: scores, self.EVAL_SCORE_DETAILED: scores_detailed,
                self.EVAL_REPORT: reports}

    def load_and_eval(self, data_loader, load_path, batch=False):
        # Load reference data and evaluate generated outputs
        reps = {}
        with gzip.open(load_path, 'rt', encoding='utf-8') as f:
            for line in f:
                entry = line.rstrip().split(' ')
                rid = entry[0].split(self.ID_SEPARATOR)[0]
                if rid not in reps:
                    reps[rid] = OrderedDict()
                reps[rid][entry[0]] = ' '.join(entry[2:])
        report_ids, reports, hypos, refs = [], [], {}, {}
        for rids, _, targ, _ in data_loader:
            for rid, reference in zip(rids, targ):
                if data_loader.dataset.multi_instance:
                    reference = reference.split(ToTokenizedTexts.INSTANCE_BREAK)
                else:
                    reference = [reference]
                # Recovered Samples
                if rid in reps:
                    for rid2, rep in reps[rid].items():
                        report_ids.append(rid2)
                        reports.append(rep)
                        hypos[rid2] = [rep]
                        refs[rid2] = []
                        for ref in reference:
                            refs[rid2].append(ref.replace('\n', ' '))
        # Calculate IDFs for NLI-TFIDF
        if self.nli is not None and SimpleNLI.COMPARE_TFIDF in self.nli_compare:
            tfidf_vectorizer = self.compute_tfidf_vectorizer(data_loader)
        else:
            tfidf_vectorizer = None
        # Evaluate with metrics
        if batch:
            scores, scores_detailed = self.eval_batch(report_ids, refs, hypos, tfidf_vectorizer,
                                                      batch_size=self.EVAL_SIZE, progress_name='eval')
        else:
            scores, scores_detailed = self.eval_batch(report_ids, refs, hypos, tfidf_vectorizer)
        return {self.EVAL_ID: report_ids, self.EVAL_SCORE: scores, self.EVAL_SCORE_DETAILED: scores_detailed,
                self.EVAL_REPORT: reports}

    def metrics(self):
        return self.get_metrics(self.bleu, self.rouge, self.cider, self.spice, self.bert_score, self.nli,
                                self.entity_match, self.nli_compare, self.qa_score, self.chexpert)

    def setup(self):
        self.qa_model = None
        if self.nli == self.NLI_MED or self.nli == self.NLI_RAD_AUG:
            bert_score = None
            for nli_comp in self.nli_compare:
                if nli_comp.startswith(SimpleNLI.COMPARE_BERT_SCORE):
                    bert_score = self.BERT_SCORE_DEFAULT
            if self.nli == self.NLI_RAD_AUG:
                resource_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources')
                model = os.path.join(resource_dir, 'model_medrad_19k')
            elif self.nli == self.NLI_MED:
                resource_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources')
                model = os.path.join(resource_dir, 'model_med')
            else:
                model = None
            model = SimpleNLI.load_model(model)
            self.nli_model = SimpleNLI(model, batch=self.nli_batch, neutral_score=self.nli_neutral_score,
                                       nthreads=self.nthreads, pin_memory=self.pin_memory, bert_score=bert_score,
                                       sentsplitter=self.sentsplitter, cache=self.nli_cache, verbose=self.verbose)
        if self.bert_score is not None:
            self.bert_score_model = BERTScorer(model_type=self.bert_score, batch_size=self.nli_batch,
                                               nthreads=self.nthreads, lang='en', rescale_with_baseline=True,
                                               penalty=self.bert_score_penalty)
        if self.qa_score is True:

            device_id = -1 if self.device == 'cpu' else 0
            self.qa_model = pipeline("question-answering", 
                                     model='/home/otabek.nazarov/Downloads/thesis/pubmed_bert_squadv2', 
                                     tokenizer='/home/otabek.nazarov/Downloads/thesis/pubmed_bert_squadv2',
                                     framework='pt', 
                                     device=device_id)

            self.bert_score_qa_model = BERTScorer(model_type=self.bert_score, 
                                                  batch_size=self.nli_batch, 
                                                  nthreads=self.nthreads, 
                                                  lang='en', 
                                                  rescale_with_baseline=True, 
                                                  penalty=self.bert_score_penalty)
            self.bert_score_qa_model.device = 'cuda:0'
            self.bert_score_qa_model.cuda(device='cuda:0')
            self.bert_score_qa_model.model.cuda(device='cuda:0')
        
        if self.entity_match is not None:
            if self.entity_mode.startswith(EntityMatcher.MODE_NLI):
                resource_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources')
                model = os.path.join(resource_dir, SimpleNLI.RADNLI_STATES)
                model = SimpleNLI.load_model(model)
                nli_model = SimpleNLI(model, batch=self.nli_batch, neutral_score=self.nli_neutral_score,
                                      nthreads=self.nthreads, pin_memory=self.pin_memory,
                                      bert_score=self.BERT_SCORE_DEFAULT, sentsplitter='linebreak',
                                      cache=self.nli_cache, verbose=self.verbose)
            else:
                nli_model = None
            target_types = {'ANATOMY': True, 'OBSERVATION': True}
            sentences, entities = EntityMatcher.load_entities(self.entity_match, target_types)
            self.entity_matcher = EntityMatcher(sentences, entities, target_types, self.entity_mode, self.nli_batch,
                                                nli_model)
