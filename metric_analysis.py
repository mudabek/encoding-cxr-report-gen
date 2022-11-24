from transformers import pipeline
from clinicgen.nli import BERTScorer
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import torch

# Set visible gpu
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]=2

# Questions for QA model
QUESTIONS = [
        'Is there pneumonia?',
        'Is there edema?',
        'Is there thorax?',
        'Are there devices?',
        'Is there opacity?',
        'Is there atelectasis?',
        'Is there cardiomegaly?',
        'Is there lung lesion?',
        'Is there consolidation?',
        'Is there fracture?',
    ]

# Model for BERTScore
bert_model = 'distilbert-base-uncased'

# Read original reports
reports_df = pd.read_csv('/home/otabek.nazarov/Downloads/thesis/ifcc/labeled_reports_test.csv')

# Batch size configuration for model
samples_cnt = 1500#3854
batch_size = 50#47 # or 41
batch_count = int(samples_cnt / batch_size)

# Load QA model
device_id = 2 # -1 for cpu
qa_model = pipeline("question-answering", 
                    model='franklu/pubmed_bert_squadv2', 
                    framework='pt',
                    device=device_id)
qa_model.model.to(torch.device('cuda:2'))

QA_THRESHOLD = 0.25

# Load BERTScore model
bert_score_qa_model = BERTScorer(model_type=bert_model, batch_size=batch_size,
                                 nthreads=2, lang='en', rescale_with_baseline=True,
                                 penalty=False)
# bert_score_qa_model.cuda()#.model.to(torch.device('cuda:2'))

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

bert_scores_detailed = []

mask_reports_dict = {}

for percent in tqdm(range(0, 100, 4)):
    # Mask out all the reports
    orig_reports = reports_df['Report Impression'].values[:samples_cnt]
    mask_reports = []
    masking_prob = percent / 100
    data_dict['mask_prob'].append(masking_prob)

    for cur_report in orig_reports:
        # Split report into list of words
        words = cur_report.split()
        words_array = np.array(words)
        length = len(words_array)

        # Mask out the words with masking_prob
        mask = np.random.choice([0, 1], size=length, replace=True, p=[1-masking_prob, masking_prob]).astype(bool)
        mask_vals = np.array([''] * length)
        words_array[mask] = mask_vals[mask]

        # Append masked report to the list
        masked_report = ' '.join(words_array.tolist())
        masked_report = re.sub(' +', ' ', masked_report)
        mask_reports.append(masked_report)

    # Save masked reports for dataframe
    mask_reports_dict[f'masked_{percent}'] = mask_reports

    # Turn into batches for fast processing
    orig_reports = np.reshape(orig_reports, (batch_count, batch_size))
    mask_reports = np.reshape(np.array(mask_reports), (batch_count, batch_size))


    f1_score_means = []
    f1_score_means_orig = []
    prec_means = []
    prec_means_orig = []
    recall_means = []
    recall_means_orig = []
    bert_scores = []
    for idx in range(batch_count):

        refs_l = orig_reports[idx,:].tolist()
        hypos_l = mask_reports[idx,:].tolist()

        f1_scores = np.empty((len(refs_l), len(QUESTIONS)))
        f1_scores.fill(np.nan)

        f1_scores_orig = np.empty((len(refs_l), len(QUESTIONS)))
        f1_scores_orig.fill(np.nan)

        prec_scores = np.empty((len(refs_l), len(QUESTIONS)))
        prec_scores.fill(np.nan)

        prec_scores_orig = np.empty((len(refs_l), len(QUESTIONS)))
        prec_scores_orig.fill(np.nan)

        recall_scores = np.empty((len(refs_l), len(QUESTIONS)))
        recall_scores.fill(np.nan)

        recall_scores_orig = np.empty((len(refs_l), len(QUESTIONS)))
        recall_scores_orig.fill(np.nan)

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
            
            b_prec, b_recall, b_f1 = bert_score_qa_model.score(bert_score_hypo, bert_score_refs)
            b_prec, b_recall, b_f1 = b_prec.numpy(), b_recall.numpy(), b_f1.numpy()

            full_prec, full_recall, full_f1 = bert_score_qa_model.score(hypos_l, refs_l)
            full_prec, full_recall, full_f1 = full_prec.numpy(), full_recall.numpy(), full_f1.numpy()

            # Select scores for loss based on threshold
            for sample_idx, (cur_ref_res, cur_hypo_res) in enumerate(zip(refs_cur_results, hypo_cur_results)):
                if cur_ref_res['score'] > QA_THRESHOLD or cur_hypo_res['score'] > QA_THRESHOLD:
                    f1_scores[sample_idx, q_idx] = b_f1[sample_idx]
                    f1_scores_orig[sample_idx, q_idx] = full_f1[sample_idx]

                    prec_scores[sample_idx, q_idx] = b_prec[sample_idx]
                    prec_scores_orig[sample_idx, q_idx] = full_prec[sample_idx]

                    recall_scores[sample_idx, q_idx] = b_recall[sample_idx]
                    recall_scores_orig[sample_idx, q_idx] = full_recall[sample_idx]

        bert_scores.append(np.nanmean(f1_scores, axis=0))
        f1_score_means.append(np.nanmean(f1_scores))
        f1_score_means_orig.append(np.nanmean(f1_scores_orig))
        prec_means.append(np.nanmean(prec_scores))
        prec_means_orig.append(np.nanmean(prec_scores_orig))
        recall_means.append(np.nanmean(recall_scores))
        recall_means_orig.append(np.nanmean(recall_scores_orig))
    
    # Save data for final dataframe
    bert_scores_detailed.append(np.array(bert_scores).mean(axis=0))
    data_dict['f1_full'].append(np.array(f1_score_means_orig).mean())
    data_dict['f1_qa'].append(np.array(f1_score_means).mean())
    data_dict['prec_full'].append(np.array(prec_means_orig).mean())
    data_dict['prec_qa'].append(np.array(prec_means).mean())
    data_dict['recall_full'].append(np.array(recall_means_orig).mean())
    data_dict['recall_qa'].append(np.array(recall_means).mean())

# Save metrics dataframe
save_df = pd.DataFrame(data_dict)
save_df.to_csv(f'metric_experiments_{QA_THRESHOLD}.csv', index=False)

# Save masked reports dataframe
save_df = pd.DataFrame(mask_reports_dict)
save_df.to_csv(f'masked_reports_{QA_THRESHOLD}.csv', index=False)

# Save detailed bert scores
bert_scores_np = np.array(bert_scores_detailed)
save_df = pd.DataFrame(bert_scores_np, columns=QUESTIONS)
save_df.to_csv(f'bert_scores_{QA_THRESHOLD}.csv', index=False)