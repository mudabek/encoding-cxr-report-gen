from transformers import pipeline
from clinicgen.nli import BERTScorer
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import torch

# Questions for QA model
QUESTIONS = [
        'Is there monia?',
        'Is there edema?',
        'Is there thorax?',
        'Are there devices?',
        'Is there opacity?',
        'Is there atelectasis?',
        'Is there heart?',
        'Is there lung lesion?',
        'Is there consolidation?',
        'Is there fracture?',
    ]

# Model for BERTScore
bert_model = 'distilbert-base-uncased'

# Read original reports
reports_df = pd.read_csv('/home/otabek.nazarov/Downloads/thesis/ifcc/labeled_reports_test.csv')
genered_df = pd.read_csv('/home/otabek.nazarov/Downloads/thesis/ifcc/trans_baseline.csv')

# Batch size configuration for model
samples_cnt = 3800
batch_size = 50#47 # or 41
batch_count = int(samples_cnt / batch_size)

# Load QA model
device_id = 2 # -1 for cpu
qa_model = pipeline("question-answering", 
                    model='franklu/pubmed_bert_squadv2', 
                    framework='pt',
                    device=device_id)
qa_model.model.to(torch.device('cuda:2'))

QA_THRESHOLD = 0.30

# Load BERTScore model
bert_score_qa_model = BERTScorer(model_type=bert_model, batch_size=batch_size,
                                 nthreads=2, lang='en', rescale_with_baseline=True,
                                 penalty=False)

# Dictionary for final dataframe
data_dict = {
    'mask_prob' : [],
    'f1_full' : [],
    'f1_qa' : [],
    'prec_full' : [],
    'prec_qa' : [],
    'recall_full' : [],
    'recall_qa' : [],
}

mask_reports_dict = {}


# Turn into batches for fast processing
orig_reports = np.reshape(reports_df['Report Impression'].values[:samples_cnt], (batch_count, batch_size))
mask_reports = np.reshape(genered_df['Report Impression'].values[:samples_cnt], (batch_count, batch_size))


f1_score_means = []
f1_score_means_orig = []
qa_bert_scores = []
full_bert_scores = []
for idx in tqdm(range(batch_count)):

    refs_l = orig_reports[idx,:].tolist()
    hypos_l = mask_reports[idx,:].tolist()

    f1_scores = np.empty((len(refs_l), len(QUESTIONS)))
    f1_scores.fill(np.nan)

    full_f1_scores = np.empty((len(refs_l), len(QUESTIONS)))
    full_f1_scores.fill(np.nan)

    for q_idx, cur_question in enumerate(QUESTIONS):    
        # Copy questions for batch forwarding to the model
        question_batch = [cur_question] * len(hypos_l)

        # Get results from QA model
        refs_cur_results = qa_model(question=question_batch, context=refs_l)
        hypo_cur_results = qa_model(question=question_batch, context=hypos_l)

        # Get bert scores for given answers
        bert_score_refs = []
        bert_score_hypo = []
        for sample_idx, (cur_ref_res, cur_hypo_res) in enumerate(zip(refs_cur_results, hypo_cur_results)):
            bert_score_refs.append(cur_ref_res['answer'])
            bert_score_hypo.append(cur_hypo_res['answer'])
        
        _, _, b_f1 = bert_score_qa_model.score(bert_score_hypo, bert_score_refs)
        b_f1 = b_f1.numpy()

        _, _, full_f1 = bert_score_qa_model.score(hypos_l, refs_l)
        full_f1 = full_f1.numpy()

        # Select scores for loss based on threshold
        for sample_idx, (cur_ref_res, cur_hypo_res) in enumerate(zip(refs_cur_results, hypo_cur_results)):
            if cur_ref_res['score'] > QA_THRESHOLD or cur_hypo_res['score'] > QA_THRESHOLD:
                f1_scores[sample_idx, q_idx] = b_f1[sample_idx]
                full_f1_scores[sample_idx, q_idx] = full_f1[sample_idx]
    
    qa_bert_scores.append(f1_scores)
    full_bert_scores.append(full_f1_scores)

# Save bert scores
bert_scores_np = np.reshape(np.array(qa_bert_scores), (samples_cnt, len(QUESTIONS)))
save_df = pd.DataFrame(bert_scores_np, columns=QUESTIONS)
save_df.to_csv(f'qa_bert_scores_heart_{QA_THRESHOLD}.csv', index=False)

bert_scores_np = np.reshape(np.array(full_bert_scores), (samples_cnt, len(QUESTIONS)))
save_df = pd.DataFrame(bert_scores_np, columns=QUESTIONS)
save_df.to_csv(f'full_bert_scores_heart_{QA_THRESHOLD}.csv', index=False)