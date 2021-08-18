# Evaluation util functions for PropBank SRL.

import codecs
from collections import Counter
import operator
import os
from os.path import join
import subprocess
import relation_metrics

from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


_SRL_CONLL_EVAL_SCRIPT  = "run_conll_eval.sh"


def split_example_for_eval(example):
  """Split document-based samples into sentence-based samples for evaluation.

  Args:
    example: 
  Returns:
    Tuple of (sentence, list of SRL relations)
  """
  sentences = example["sentences"]
  if "srl" not in example:
    example["srl"] = [[] for s in sentences]
  if "relations" not in example:
    example["relations"] = [[] for s in sentences]
  num_words = sum(len(s) for s in sentences)
  word_offset = 0
  samples = []
  for i, sentence in enumerate(sentences):
    srl_rels = {}
    ner_spans = []
    stat_spans = []
    relations = []
    for r in example["srl"][i]:
      pred_id = r[0] - word_offset
      if pred_id not in srl_rels:
        srl_rels[pred_id] = []
      srl_rels[pred_id].append((r[1] - word_offset, r[2] - word_offset, r[3]))
    for r in example["ner"][i]:
      ner_spans.append((r[0] - word_offset, r[1] - word_offset, r[2]))
    for r in example["stat"][i]:
      stat_spans.append((r[0] - word_offset, r[1] - word_offset, r[2]))
    for r in example["relations"][i]:
      relations.append((r[0] - word_offset, r[1] - word_offset, r[2] - word_offset,
                        r[3] - word_offset, r[4]))
    samples.append((sentence, srl_rels, ner_spans, relations, stat_spans))
    word_offset += len(sentence)
  return samples


def evaluate_retrieval(span_starts,span_ends,span_scores,pred_starts, 
                      pred_ends, gold_spans,
                       text_length, evaluators, debugging=False):
  """
  Evaluation for unlabeled retrieval.

  Args:
    gold_spans: Set of tuples of (start, end).
  """
  if len(span_starts) > 0:
    sorted_starts, sorted_ends, sorted_scores = zip(*sorted(
        zip(span_starts, span_ends, span_scores),
        key=operator.itemgetter(2), reverse=True))
  else:
    sorted_starts = []
    sorted_ends = []
  for k, evaluator in evaluators.items():
    if k == -3:
      predicted_spans = set(zip(span_starts, span_ends)) & gold_spans
    else:
      if k == -2:
        predicted_starts = pred_starts
        predicted_ends = pred_ends
        if debugging:
          print "Predicted", zip(sorted_starts, sorted_ends, sorted_scores)[:len(gold_spans)]
          print "Gold", gold_spans
     # FIXME: scalar index error
      elif k == 0:
        is_predicted = span_scores > 0
        predicted_starts = span_starts[is_predicted]
        predicted_ends = span_ends[is_predicted]
      else:
        if k == -1:
          num_predictions = len(gold_spans)
        else:
          num_predictions = (k * text_length) / 100
        predicted_starts = sorted_starts[:num_predictions]
        predicted_ends = sorted_ends[:num_predictions]
      predicted_spans = set(zip(predicted_starts, predicted_ends))
    evaluator.update(gold_set=gold_spans, predicted_set=predicted_spans)


def _print_f1(total_gold, total_predicted, total_matched, message=""):
  precision = 100.0 * total_matched / total_predicted if total_predicted > 0 else 0
  recall = 100.0 * total_matched / total_gold if total_gold > 0 else 0
  f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
  print ("{}: Precision: {}, Recall: {}, F1: {}".format(message, precision, recall, f1))
  return precision, recall, f1


def compute_span_f1(gold_data, predictions, task_name):
  print('len(gold_data)',len(gold_data))
  print('len(predictions)',len(predictions))
  print('gold_data',gold_data)
  print('predictions',predictions)
  assert len(gold_data) == len(predictions)
  total_gold = 0
  total_predicted = 0
  total_matched = 0
  total_unlabeled_matched = 0
  label_confusions = Counter()  # Counter of (gold, pred) label pairs.

  pd_m = []
  gd_m = []
  lab_dict = {'symptom':0,'medication':1,'otherMedicalCondition':2,'procedure':4}
  # lab_dict = {"pat-pos":0, "pat-neg":1,  "doc-pos":2, "doc-neg":3}
  for i in range(len(gold_data)):
    gold = gold_data[i]
    pred = predictions[i]
    if gold and pred:
      # pd_m = []
      # gd_m = []
    # total_gold += len(gold)
    # total_predicted += len(pred)
      for id_gd in range(len(gold)):
        a0 = gold[id_gd]
        a0_lab = a0[2].encode("utf-8")
        print('a0_lab',a0_lab)
        for id_pd in range(len(pred)):
          a1 = pred[id_pd]
          a1_lab = a1[2].encode("utf-8")
          print('a1_lab',a1_lab)
          if a0_lab != 'property':
            if a1_lab != 'property':
              print('here a0_lab',a0_lab)
              print('here a1_lab',a1_lab)
              if a0_lab == 'otherMedicalCondition' and a1_lab == 'symptom':
                pd_m.append(lab_dict[a0_lab])
                gd_m.append(lab_dict[a0_lab])
              # elif a0_lab == 'otherMedicalCondition' and a1_lab == 'procedure':
              #   pd_m.append(lab_dict[a0_lab])
              #   gd_m.append(lab_dict[a0_lab])
              else:
                pd_m.append(lab_dict[a1_lab])
                gd_m.append(lab_dict[a0_lab])
          # pd_m.append(lab_dict[a1_lab])
          # gd_m.append(lab_dict[a0_lab])
            
  
  print('len(gd_m)',len(gd_m))
  print('gd_m',gd_m)
  print('len(pd_m)',len(pd_m))
  print('pd_m',pd_m)

  target_names = ["symptom", "medication", "otherMedicalCondition", "procedure"]
  # target_names = ["pat-pos", "pat-neg",  "doc-pos", "doc-neg"]
  cr = classification_report(gd_m, pd_m, target_names=target_names)
  print('classification report',cr)

  # print('type(gold_data)',type(gold_data))
  # print('len(gold_data)',len(gold_data))
  # print('gold_data',gold_data)
  # print('type(predictions)',type(predictions))
  # print('len(predictions)',len(predictions))
  # print('predictions',predictions)

  # sys.exit()
  mac_prec_st = []
  mac_rec_st = []
  mac_f1_st = []

  mic_prec_st = []
  mic_rec_st = []
  mic_f1_st = []

  wei_prec_st = []
  wei_rec_st = []
  wei_f1_st = []

  mac_prec_ed = []
  mac_rec_ed = []
  mac_f1_ed = []

  mic_prec_ed = []
  mic_rec_ed = []
  mic_f1_ed = []

  wei_prec_ed = []
  wei_rec_ed = []
  wei_f1_ed = []

  mac_prec_lab = []
  mac_rec_lab = []
  mac_f1_lab = []

  mic_prec_lab = []
  mic_rec_lab = []
  mic_f1_lab = []

  wei_prec_lab = []
  wei_rec_lab = []
  wei_f1_lab = []


  mac_prec_m = []
  mac_rec_m = []
  mac_f1_m = []

  mic_prec_m = []
  mic_rec_m = []
  mic_f1_m = []

  wei_prec_m = []
  wei_rec_m = []
  wei_f1_m = []


  mac_prec_um = []
  mac_rec_um = []
  mac_f1_um = []

  mic_prec_um = []
  mic_rec_um = []
  mic_f1_um = []

  wei_prec_um = []
  wei_rec_um = []
  wei_f1_um = []

  accu_m = []
  accu_um = []

  for i in range(len(gold_data)):
    gold = gold_data[i]
    pred = predictions[i]
    # print('len(gold)',len(gold))
    # print('gold',gold)
    # print('len(pred)',len(pred))
    # print('pred',pred)

    ## my part using sklearn
    gd_start_lis = []
    gd_end_lis = []
    gd_lab_lis = []
    pd_start_lis = []
    pd_end_lis = []
    pd_lab_lis = []
    st_ed_lis = []

    if gold and pred:
      # for id_new_new in range(len(gold)):
      #   a0 = gold[id_new_new]
      #   gd_start_lis.append(a0[0])
      #   gd_end_lis.append(a0[1])
      #   gd_lab_lis.append(a0[2])

      # for a1 in pred:
      #   pd_start_lis.append(a1[0])
      #   pd_end_lis.append(a1[1])
      #   pd_lab_lis.append(a1[2])
    
      # print('len(gd_start_lis)',len(gd_start_lis))
      # print('gd_start_lis',gd_start_lis)
      # print('len(pd_start_lis)',len(pd_start_lis))
      # print('pd_start_lis',pd_start_lis)

      # precisions_mac_st, recall_mac_st, f1_score_mac_st, _ = precision_recall_fscore_support(gd_start_lis, pd_start_lis, average='macro')
      # precisions_mic_st, recall_mic_st, f1_score_mic_st, _ = precision_recall_fscore_support(gd_start_lis, pd_start_lis, average='micro')
      # precisions_wei_st, recall_wei_st, f1_score_wei_st, _ = precision_recall_fscore_support(gd_start_lis, pd_start_lis, average='weighted')
      # mac_prec_st.append(precisions_mac_st)
      # mac_rec_st.append(recall_mac_st)
      # mac_f1_st.append(f1_score_mac_st)
      # mic_prec_st.append(precisions_mic_st)
      # mic_rec_st.append(recall_mic_st)
      # mic_f1_st.append(f1_score_mic_st)
      # wei_prec_st.append(precisions_wei_st)
      # wei_rec_st.append(recall_wei_st)
      # wei_f1_st.append(f1_score_wei_st)


      # precisions_mac_ed, recall_mac_ed, f1_score_mac_ed, _ = precision_recall_fscore_support(gd_end_lis, pd_end_lis, average='macro')
      # precisions_mic_ed, recall_mic_ed, f1_score_mic_ed, _ = precision_recall_fscore_support(gd_end_lis, pd_end_lis, average='micro')
      # precisions_wei_ed, recall_wei_ed, f1_score_wei_ed, _ = precision_recall_fscore_support(gd_end_lis, pd_end_lis, average='weighted')
      # mac_prec_ed.append(precisions_mac_ed)
      # mac_rec_ed.append(recall_mac_ed)
      # mac_f1_ed.append(f1_score_mac_ed)
      # mic_prec_ed.append(precisions_mic_ed)
      # mic_rec_ed.append(recall_mic_ed)
      # mic_f1_ed.append(f1_score_mic_ed)
      # wei_prec_ed.append(precisions_wei_ed)
      # wei_rec_ed.append(recall_wei_ed)
      # wei_f1_ed.append(f1_score_wei_ed)


      # precisions_mac_lab, recall_mac_lab, f1_score_mac_lab, _ = precision_recall_fscore_support(gd_lab_lis, pd_lab_lis, average='macro')
      # precisions_mic_lab, recall_mic_lab, f1_score_mic_lab, _ = precision_recall_fscore_support(gd_lab_lis, pd_lab_lis, average='micro')
      # precisions_wei_lab, recall_wei_lab, f1_score_wei_lab, _ = precision_recall_fscore_support(gd_lab_lis, pd_lab_lis, average='weighted')
      # mac_prec_lab.append(precisions_mac_lab)
      # mac_rec_lab.append(recall_mac_lab)
      # mac_f1_lab.append(f1_score_mac_lab)
      # mic_prec_lab.append(precisions_mic_lab)
      # mic_rec_lab.append(recall_mic_lab)
      # mic_f1_lab.append(f1_score_mic_lab)
      # wei_prec_lab.append(precisions_wei_lab)
      # wei_rec_lab.append(recall_wei_lab)
      # wei_f1_lab.append(f1_score_wei_lab)

      # gd_um = [1]*len(gold)
      pd_um = []
      # gd_m = [1]*len(gold)
      pd_m = []
      # for a0 in gold:
      # print('len(gold)',len(gold))
      # print('gold',gold)
      # print('len(pred)',len(pred))
      # print('pred',pred)
      for id_gd in range(len(gold)):
        a0 = gold[id_gd]
        for id_pd in range(len(pred)):
          a1 = pred[id_pd]
          if a0[0] == a1[0] and a0[1] == a1[1]:
            pd_um.append(1)
            if a0[2] == a1[2]:
              pd_m.append(1)
              break
            else:
              pd_m.append(0)
              break
          elif a0[0] == a1[0] and a0[1] != a1[1]:
            pd_um.append(1)
            if a0[2] == a1[2]:
              pd_m.append(1)
              break
            else:
              pd_m.append(0)
              break
          elif a0[0] != a1[0] and a0[1] == a1[1]:
            pd_um.append(1)
            if a0[2] == a1[2]:
              pd_m.append(1)
              break
            else:
              pd_m.append(0)
              break
          else:
            if id_pd == (len(pred) -1):
              pd_m.append(0)
              pd_um.append(0)

      # print('len(pd_m)',len(pd_m))
      # print('pd_m',pd_m)
      # print('len(pd_um)',len(pd_um))
      # print('pd_um',pd_um)
      gd_um = [1]*len(pd_um)
      # print('len(gd_um)',len(gd_um))
      # print('gd_um',gd_um)
      gd_m = [1]*len(pd_m)
      # print('len(gd_m)',len(gd_m))
      # print('gd_m',gd_m)

      precisions_mac_m, recall_mac_m, f1_score_mac_m, _ = precision_recall_fscore_support(gd_m, pd_m, average='macro')
      precisions_mic_m, recall_mic_m, f1_score_mic_m, _ = precision_recall_fscore_support(gd_m, pd_m, average='micro')
      precisions_wei_m, recall_wei_m, f1_score_wei_m, _ = precision_recall_fscore_support(gd_m, pd_m, average='weighted')
      mac_prec_m.append(precisions_mac_m)
      mac_rec_m.append(recall_mac_m)
      mac_f1_m.append(f1_score_mac_m)
      mic_prec_m.append(precisions_mic_m)
      mic_rec_m.append(recall_mic_m)
      mic_f1_m.append(f1_score_mic_m)
      wei_prec_m.append(precisions_wei_m)
      wei_rec_m.append(recall_wei_m)
      wei_f1_m.append(f1_score_wei_m)
      accuracy_m = accuracy_score(gd_m, pd_m)
      accu_m.append(accuracy_m)


      precisions_mac_um, recall_mac_um, f1_score_mac_um, _ = precision_recall_fscore_support(gd_um, pd_um, average='macro')
      precisions_mic_um, recall_mic_um, f1_score_mic_um, _ = precision_recall_fscore_support(gd_um, pd_um, average='micro')
      precisions_wei_um, recall_wei_um, f1_score_wei_um, _ = precision_recall_fscore_support(gd_um, pd_um, average='weighted')
      mac_prec_um.append(precisions_mac_um)
      mac_rec_um.append(recall_mac_um)
      mac_f1_um.append(f1_score_mac_um)
      mic_prec_um.append(precisions_mic_um)
      mic_rec_um.append(recall_mic_um)
      mic_f1_um.append(f1_score_mic_um)
      wei_prec_um.append(precisions_wei_um)
      wei_rec_um.append(recall_wei_um)
      wei_f1_um.append(f1_score_wei_um)
      accuracy_um = accuracy_score(gd_um, pd_um)
      accu_um.append(accuracy_um)


  #####################################################################

    # gold = gold_data[i]
    # pred = predictions[i]
    total_gold += len(gold)
    total_predicted += len(pred)
    
    for a0 in gold:
      for a1 in pred:
        if a0[0] == a1[0] and a0[1] == a1[1]:
          total_unlabeled_matched += 1
          label_confusions.update([(a0[2], a1[2]),])
          if a0[2] == a1[2]:
            total_matched += 1

  prec, recall, f1 = _print_f1(total_gold, total_predicted, total_matched, task_name)
  ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_predicted, total_unlabeled_matched, "Unlabeled " + task_name)
  
  
  
  ######################################################################
  ## my code
  # gd_um = [1]*len(gold)
  # pd_um = []
  # gd_m = [1]*len(gold)
  # pd_m = []
  # for id_new in range(len(gold)):
  #   a0 = gold[id_new]
  #   a1 = pred[id_new]
  #   if a0[0] == a1[0] and a0[1] == a1[1]:
  #     pd_um.append(1)
  #     if a0[2] == a1[2]:
  #       pd_m.append(1)
  #     else:
  #       pd_m.append(0)
  #   else:
  #     pd_um.append(0)


  # precisions_mac_m, recall_mac_m, f1_score_mac_m, _ = precision_recall_fscore_support(gd_m, pd_m, average='macro')
  # precisions_mic_m, recall_mic_m, f1_score_mic_m, _ = precision_recall_fscore_support(gd_m, pd_m, average='micro')
  # precisions_wei_m, recall_wei_m, f1_score_wei_m, _ = precision_recall_fscore_support(gd_m, pd_m, average='weighted')
  # mac_prec_m.append(precisions_mac_m)
  # mac_rec_m.append(recall_mac_m)
  # mac_f1_m.append(f1_score_mac_m)
  # mic_prec_m.append(precisions_mic_m)
  # mic_rec_m.append(recall_mic_m)
  # mic_f1_m.append(f1_score_mic_m)
  # wei_prec_m.append(precisions_wei_m)
  # wei_rec_m.append(recall_wei_m)
  # wei_f1_m.append(f1_score_wei_m)

  print('MATCHED')
  print('mac scores')
  score_avg_prec_m_mac = sum(mac_prec_m) / len(mac_prec_m)
  print('score_avg_prec_m_mac', score_avg_prec_m_mac)
  
  score_avg_rec_m_mac = sum(mac_rec_m) / len(mac_rec_m)
  print('score_avg_recall_m_mac', score_avg_rec_m_mac)
  
  score_avg_f1_m_mac = sum(mac_f1_m) / len(mac_f1_m)
  print('score_avg_f1_m_mac', score_avg_f1_m_mac)

  print('mic scores')
  score_avg_prec_m_mic = sum(mic_prec_m) / len(mic_prec_m)
  print('score_avg_prec_m_mic', score_avg_prec_m_mic)
  
  score_avg_rec_m_mic = sum(mic_rec_m) / len(mic_rec_m)
  print('score_avg_recall_m_mic', score_avg_rec_m_mic)
  
  score_avg_f1_m_mic = sum(mic_f1_m) / len(mic_f1_m)
  print('score_avg_f1_m_mic', score_avg_f1_m_mic)


  print('wei scores')
  score_avg_prec_m_wei = sum(wei_prec_m) / len(wei_prec_m)
  print('score_avg_prec_m_wei', score_avg_prec_m_wei)
  
  score_avg_rec_m_wei = sum(wei_rec_m) / len(wei_rec_m)
  print('score_avg_recall_m_wei', score_avg_rec_m_wei)
  
  score_avg_f1_m_wei = sum(wei_f1_m) / len(wei_f1_m)
  print('score_avg_f1_m_wei', score_avg_f1_m_wei)


  score_avg_accu_m = sum(accu_m) / len(accu_m)
  print('score_avg_accu_m', score_avg_accu_m)


  # precisions_mac_um, recall_mac_um, f1_score_mac_um, _ = precision_recall_fscore_support(gd_um, pd_um, average='macro')
  # precisions_mic_um, recall_mic_um, f1_score_mic_um, _ = precision_recall_fscore_support(gd_um, pd_um, average='micro')
  # precisions_wei_um, recall_wei_um, f1_score_wei_um, _ = precision_recall_fscore_support(gd_um, pd_um, average='weighted')
  # mac_prec_um.append(precisions_mac_um)
  # mac_rec_um.append(recall_mac_um)
  # mac_f1_um.append(f1_score_mac_um)
  # mic_prec_um.append(precisions_mic_um)
  # mic_rec_um.append(recall_mic_um)
  # mic_f1_um.append(f1_score_mic_um)
  # wei_prec_um.append(precisions_wei_um)
  # wei_rec_um.append(recall_wei_um)
  # wei_f1_um.append(f1_score_wei_um)

  print('UNMATCHED')
  print('mac scores')
  score_avg_prec_um_mac = sum(mac_prec_um) / len(mac_prec_um)
  print('score_avg_prec_um_mac', score_avg_prec_um_mac)
  
  score_avg_rec_um_mac = sum(mac_rec_um) / len(mac_rec_um)
  print('score_avg_recall_um_mac', score_avg_rec_um_mac)
  
  score_avg_f1_um_mac = sum(mac_f1_um) / len(mac_f1_um)
  print('score_avg_f1_um_mac', score_avg_f1_um_mac)

  print('mic scores')
  score_avg_prec_um_mic = sum(mic_prec_um) / len(mic_prec_um)
  print('score_avg_prec_um_mic', score_avg_prec_um_mic)
  
  score_avg_rec_um_mic = sum(mic_rec_um) / len(mic_rec_um)
  print('score_avg_recall_um_mic', score_avg_rec_um_mic)
  
  score_avg_f1_um_mic = sum(mic_f1_um) / len(mic_f1_um)
  print('score_avg_f1_um_mic', score_avg_f1_um_mic)


  print('wei scores')
  score_avg_prec_um_wei = sum(wei_prec_um) / len(wei_prec_um)
  print('score_avg_prec_um_wei', score_avg_prec_um_wei)
  
  score_avg_rec_um_wei = sum(wei_rec_um) / len(wei_rec_um)
  print('score_avg_recall_um_wei', score_avg_rec_um_wei)
  
  score_avg_f1_um_wei = sum(wei_f1_um) / len(wei_f1_um)
  print('score_avg_f1_um_wei', score_avg_f1_um_wei)

  score_avg_accu_um = sum(accu_um) / len(accu_um)
  print('score_avg_accu_um', score_avg_accu_um)



  # print('START')
  # print('mac scores')
  # score_avg_prec_st_mac = sum(mac_prec_st) / len(mac_prec_st)
  # print('score_avg_prec_st_mac', score_avg_prec_st_mac)
  
  # score_avg_rec_st_mac = sum(mac_rec_st) / len(mac_rec_st)
  # print('score_avg_recall_st_mac', score_avg_rec_st_mac)
  
  # score_avg_f1_st_mac = sum(mac_f1_st) / len(mac_f1_st)
  # print('score_avg_f1_st_mac', score_avg_f1_st_mac)

  # print('mic scores')
  # score_avg_prec_st_mic = sum(mic_prec_st) / len(mic_prec_st)
  # print('score_avg_prec_st_mic', score_avg_prec_st_mic)
  
  # score_avg_rec_st_mic = sum(mic_rec_st) / len(mic_rec_st)
  # print('score_avg_recall_st_mic', score_avg_rec_st_mic)
  
  # score_avg_f1_st_mic = sum(mic_f1_st) / len(mic_f1_st)
  # print('score_avg_f1_st_mic', score_avg_f1_st_mic)


  # print('wei scores')
  # score_avg_prec_st_wei = sum(wei_prec_st) / len(wei_prec_st)
  # print('score_avg_prec_st_wei', score_avg_prec_st_wei)
  
  # score_avg_rec_st_wei = sum(wei_rec_st) / len(wei_rec_st)
  # print('score_avg_recall_st_wei', score_avg_rec_st_wei)
  
  # score_avg_f1_st_wei = sum(wei_f1_st) / len(wei_f1_st)
  # print('score_avg_f1_st_wei', score_avg_f1_st_wei)



  # print('END')
  # print('mac scores')
  # score_avg_prec_ed_mac = sum(mac_prec_ed) / len(mac_prec_ed)
  # print('score_avg_prec_ed_mac', score_avg_prec_ed_mac)
  
  # score_avg_rec_ed_mac = sum(mac_rec_ed) / len(mac_rec_ed)
  # print('score_avg_recall_ed_mac', score_avg_rec_ed_mac)
  
  # score_avg_f1_ed_mac = sum(mac_f1_ed) / len(mac_f1_ed)
  # print('score_avg_f1_ed_mac', score_avg_f1_ed_mac)

  # print('mic scores')
  # score_avg_prec_ed_mic = sum(mic_prec_ed) / len(mic_prec_ed)
  # print('score_avg_prec_ed_mic', score_avg_prec_ed_mic)
  
  # score_avg_rec_ed_mic = sum(mic_rec_ed) / len(mic_rec_ed)
  # print('score_avg_recall_ed_mic', score_avg_rec_ed_mic)
  
  # score_avg_f1_ed_mic = sum(mic_f1_ed) / len(mic_f1_ed)
  # print('score_avg_f1_ed_mic', score_avg_f1_ed_mic)


  # print('wei scores')
  # score_avg_prec_ed_wei = sum(wei_prec_ed) / len(wei_prec_ed)
  # print('score_avg_prec_ed_wei', score_avg_prec_ed_wei)
  
  # score_avg_rec_ed_wei = sum(wei_rec_ed) / len(wei_rec_ed)
  # print('score_avg_recall_ed_wei', score_avg_rec_ed_wei)
  
  # score_avg_f1_ed_wei = sum(wei_f1_ed) / len(wei_f1_ed)
  # print('score_avg_f1_ed_wei', score_avg_f1_ed_wei)



  # print('LAB')
  # print('mac scores')
  # score_avg_prec_lab_mac = sum(mac_prec_lab) / len(mac_prec_lab)
  # print('score_avg_prec_lab_mac', score_avg_prec_lab_mac)
  
  # score_avg_rec_lab_mac = sum(mac_rec_lab) / len(mac_rec_lab)
  # print('score_avg_recall_lab_mac', score_avg_rec_lab_mac)
  
  # score_avg_f1_lab_mac = sum(mac_f1_lab) / len(mac_f1_lab)
  # print('score_avg_f1_lab_mac', score_avg_f1_lab_mac)

  # print('mic scores')
  # score_avg_prec_lab_mic = sum(mic_prec_lab) / len(mic_prec_lab)
  # print('score_avg_prec_lab_mic', score_avg_prec_lab_mic)
  
  # score_avg_rec_lab_mic = sum(mic_rec_lab) / len(mic_rec_lab)
  # print('score_avg_recall_lab_mic', score_avg_rec_lab_mic)
  
  # score_avg_f1_lab_mic = sum(mic_f1_lab) / len(mic_f1_lab)
  # print('score_avg_f1_lab_mic', score_avg_f1_lab_mic)


  # print('wei scores')
  # score_avg_prec_lab_wei = sum(wei_prec_lab) / len(wei_prec_lab)
  # print('score_avg_prec_lab_wei', score_avg_prec_lab_wei)
  
  # score_avg_rec_lab_wei = sum(wei_rec_lab) / len(wei_rec_lab)
  # print('score_avg_recall_lab_wei', score_avg_rec_lab_wei)
  
  # score_avg_f1_lab_wei = sum(wei_f1_lab) / len(wei_f1_lab)
  # print('score_avg_f1_lab_wei', score_avg_f1_lab_wei)
  
  
  return prec, recall, f1, ul_prec, ul_recall, ul_f1, label_confusions



def compute_span_f1_stat(gold_data, predictions, task_name):
  print('len(gold_data)',len(gold_data))
  print('len(predictions)',len(predictions))
  print('gold_data',gold_data)
  print('predictions',predictions)
  assert len(gold_data) == len(predictions)
  total_gold = 0
  total_predicted = 0
  total_matched = 0
  total_unlabeled_matched = 0
  label_confusions = Counter()  # Counter of (gold, pred) label pairs.

  pd_m = []
  gd_m = []
  # lab_dict = {'symptom':0,'medication':1,'otherMedicalCondition':2,'procedure':4}
  lab_dict = {"pat-pos":0, "pat-neg":1,  "doc-pos":2, "doc-neg":3}
  s_list = ['pat-neut','doc-neut']
  for i in range(len(gold_data)):
    gold = gold_data[i]
    pred = predictions[i]
    if gold and pred:
      # pd_m = []
      # gd_m = []
    # total_gold += len(gold)
    # total_predicted += len(pred)
      for id_gd in range(len(gold)):
        a0 = gold[id_gd]
        a0_lab = a0[2].encode("utf-8")
        print('a0_lab',a0_lab)
        for id_pd in range(len(pred)):
          a1 = pred[id_pd]
          a1_lab = a1[2].encode("utf-8")
          print('a1_lab',a1_lab)
          # if a1_lab == 'pat-neg':
          #   print('1 jo')
          #     # pass
          #   pd_m.append(lab_dict['pat-neg'])
          #   gd_m.append(lab_dict['pat-neg'])
          # elif a1_lab == 'doc-neg':
          #   print('2 jo')
          #     # pass
          #   pd_m.append(lab_dict['doc-neg'])
          #   gd_m.append(lab_dict['doc-neg'])
          # elif a0_lab == 'pat-neut':
          #   pass
          # elif a0_lab == 'doc-neut':
          #   pass

          # elif a0_lab != 'pat-neut' or a0_lab != 'doc-neut':
          #   print('1 here a0_lab',a0_lab)
          #   print('1 here a1_lab',a1_lab)
          #   pd_m.append(lab_dict[a1_lab])
          #   gd_m.append(lab_dict[a0_lab])
          # elif a1_lab != 'pat-neut' or a1_lab != 'doc-neut':
          
          # else:
          #   print('2 here a0_lab',a0_lab)
          #   print('2 here a1_lab',a1_lab)
          #   pd_m.append(lab_dict[a1_lab])
          #   gd_m.append(lab_dict[a0_lab])

          print('2 here a0_lab',a0_lab)
          print('2 here a1_lab',a1_lab)
          pd_m.append(lab_dict[a1_lab])
          gd_m.append(lab_dict[a0_lab])
          # pd_m.append(lab_dict[a1_lab])
          # gd_m.append(lab_dict[a0_lab])
            
  
  print('len(gd_m)',len(gd_m))
  print('gd_m',gd_m)
  print('len(pd_m)',len(pd_m))
  print('pd_m',pd_m)

  # target_names = ["symptom", "medication", "otherMedicalCondition", "procedure"]
  target_names = ["pat-pos", "pat-neg",  "doc-pos", "doc-neg"]
  cr = classification_report(gd_m, pd_m, target_names=target_names)
  print('classification report',cr)

  # print('type(gold_data)',type(gold_data))
  # print('len(gold_data)',len(gold_data))
  # print('gold_data',gold_data)
  # print('type(predictions)',type(predictions))
  # print('len(predictions)',len(predictions))
  # print('predictions',predictions)

  # sys.exit()
  mac_prec_st = []
  mac_rec_st = []
  mac_f1_st = []

  mic_prec_st = []
  mic_rec_st = []
  mic_f1_st = []

  wei_prec_st = []
  wei_rec_st = []
  wei_f1_st = []

  mac_prec_ed = []
  mac_rec_ed = []
  mac_f1_ed = []

  mic_prec_ed = []
  mic_rec_ed = []
  mic_f1_ed = []

  wei_prec_ed = []
  wei_rec_ed = []
  wei_f1_ed = []

  mac_prec_lab = []
  mac_rec_lab = []
  mac_f1_lab = []

  mic_prec_lab = []
  mic_rec_lab = []
  mic_f1_lab = []

  wei_prec_lab = []
  wei_rec_lab = []
  wei_f1_lab = []


  mac_prec_m = []
  mac_rec_m = []
  mac_f1_m = []

  mic_prec_m = []
  mic_rec_m = []
  mic_f1_m = []

  wei_prec_m = []
  wei_rec_m = []
  wei_f1_m = []


  mac_prec_um = []
  mac_rec_um = []
  mac_f1_um = []

  mic_prec_um = []
  mic_rec_um = []
  mic_f1_um = []

  wei_prec_um = []
  wei_rec_um = []
  wei_f1_um = []

  accu_m = []
  accu_um = []

  for i in range(len(gold_data)):
    gold = gold_data[i]
    pred = predictions[i]
    # print('len(gold)',len(gold))
    # print('gold',gold)
    # print('len(pred)',len(pred))
    # print('pred',pred)

    ## my part using sklearn
    gd_start_lis = []
    gd_end_lis = []
    gd_lab_lis = []
    pd_start_lis = []
    pd_end_lis = []
    pd_lab_lis = []
    st_ed_lis = []

    if gold and pred:
      # for id_new_new in range(len(gold)):
      #   a0 = gold[id_new_new]
      #   gd_start_lis.append(a0[0])
      #   gd_end_lis.append(a0[1])
      #   gd_lab_lis.append(a0[2])

      # for a1 in pred:
      #   pd_start_lis.append(a1[0])
      #   pd_end_lis.append(a1[1])
      #   pd_lab_lis.append(a1[2])
    
      # print('len(gd_start_lis)',len(gd_start_lis))
      # print('gd_start_lis',gd_start_lis)
      # print('len(pd_start_lis)',len(pd_start_lis))
      # print('pd_start_lis',pd_start_lis)

      # precisions_mac_st, recall_mac_st, f1_score_mac_st, _ = precision_recall_fscore_support(gd_start_lis, pd_start_lis, average='macro')
      # precisions_mic_st, recall_mic_st, f1_score_mic_st, _ = precision_recall_fscore_support(gd_start_lis, pd_start_lis, average='micro')
      # precisions_wei_st, recall_wei_st, f1_score_wei_st, _ = precision_recall_fscore_support(gd_start_lis, pd_start_lis, average='weighted')
      # mac_prec_st.append(precisions_mac_st)
      # mac_rec_st.append(recall_mac_st)
      # mac_f1_st.append(f1_score_mac_st)
      # mic_prec_st.append(precisions_mic_st)
      # mic_rec_st.append(recall_mic_st)
      # mic_f1_st.append(f1_score_mic_st)
      # wei_prec_st.append(precisions_wei_st)
      # wei_rec_st.append(recall_wei_st)
      # wei_f1_st.append(f1_score_wei_st)


      # precisions_mac_ed, recall_mac_ed, f1_score_mac_ed, _ = precision_recall_fscore_support(gd_end_lis, pd_end_lis, average='macro')
      # precisions_mic_ed, recall_mic_ed, f1_score_mic_ed, _ = precision_recall_fscore_support(gd_end_lis, pd_end_lis, average='micro')
      # precisions_wei_ed, recall_wei_ed, f1_score_wei_ed, _ = precision_recall_fscore_support(gd_end_lis, pd_end_lis, average='weighted')
      # mac_prec_ed.append(precisions_mac_ed)
      # mac_rec_ed.append(recall_mac_ed)
      # mac_f1_ed.append(f1_score_mac_ed)
      # mic_prec_ed.append(precisions_mic_ed)
      # mic_rec_ed.append(recall_mic_ed)
      # mic_f1_ed.append(f1_score_mic_ed)
      # wei_prec_ed.append(precisions_wei_ed)
      # wei_rec_ed.append(recall_wei_ed)
      # wei_f1_ed.append(f1_score_wei_ed)


      # precisions_mac_lab, recall_mac_lab, f1_score_mac_lab, _ = precision_recall_fscore_support(gd_lab_lis, pd_lab_lis, average='macro')
      # precisions_mic_lab, recall_mic_lab, f1_score_mic_lab, _ = precision_recall_fscore_support(gd_lab_lis, pd_lab_lis, average='micro')
      # precisions_wei_lab, recall_wei_lab, f1_score_wei_lab, _ = precision_recall_fscore_support(gd_lab_lis, pd_lab_lis, average='weighted')
      # mac_prec_lab.append(precisions_mac_lab)
      # mac_rec_lab.append(recall_mac_lab)
      # mac_f1_lab.append(f1_score_mac_lab)
      # mic_prec_lab.append(precisions_mic_lab)
      # mic_rec_lab.append(recall_mic_lab)
      # mic_f1_lab.append(f1_score_mic_lab)
      # wei_prec_lab.append(precisions_wei_lab)
      # wei_rec_lab.append(recall_wei_lab)
      # wei_f1_lab.append(f1_score_wei_lab)

      # gd_um = [1]*len(gold)
      pd_um = []
      # gd_m = [1]*len(gold)
      pd_m = []
      # for a0 in gold:
      # print('len(gold)',len(gold))
      # print('gold',gold)
      # print('len(pred)',len(pred))
      # print('pred',pred)
      for id_gd in range(len(gold)):
        a0 = gold[id_gd]
        for id_pd in range(len(pred)):
          a1 = pred[id_pd]
          if a0[0] == a1[0] and a0[1] == a1[1]:
            pd_um.append(1)
            if a0[2] == a1[2]:
              pd_m.append(1)
              break
            else:
              pd_m.append(0)
              break
          elif a0[0] == a1[0] and a0[1] != a1[1]:
            pd_um.append(1)
            if a0[2] == a1[2]:
              pd_m.append(1)
              break
            else:
              pd_m.append(0)
              break
          elif a0[0] != a1[0] and a0[1] == a1[1]:
            pd_um.append(1)
            if a0[2] == a1[2]:
              pd_m.append(1)
              break
            else:
              pd_m.append(0)
              break
          else:
            if id_pd == (len(pred) -1):
              pd_m.append(0)
              pd_um.append(0)

      # print('len(pd_m)',len(pd_m))
      # print('pd_m',pd_m)
      # print('len(pd_um)',len(pd_um))
      # print('pd_um',pd_um)
      gd_um = [1]*len(pd_um)
      # print('len(gd_um)',len(gd_um))
      # print('gd_um',gd_um)
      gd_m = [1]*len(pd_m)
      # print('len(gd_m)',len(gd_m))
      # print('gd_m',gd_m)

      precisions_mac_m, recall_mac_m, f1_score_mac_m, _ = precision_recall_fscore_support(gd_m, pd_m, average='macro')
      precisions_mic_m, recall_mic_m, f1_score_mic_m, _ = precision_recall_fscore_support(gd_m, pd_m, average='micro')
      precisions_wei_m, recall_wei_m, f1_score_wei_m, _ = precision_recall_fscore_support(gd_m, pd_m, average='weighted')
      mac_prec_m.append(precisions_mac_m)
      mac_rec_m.append(recall_mac_m)
      mac_f1_m.append(f1_score_mac_m)
      mic_prec_m.append(precisions_mic_m)
      mic_rec_m.append(recall_mic_m)
      mic_f1_m.append(f1_score_mic_m)
      wei_prec_m.append(precisions_wei_m)
      wei_rec_m.append(recall_wei_m)
      wei_f1_m.append(f1_score_wei_m)
      accuracy_m = accuracy_score(gd_m, pd_m)
      accu_m.append(accuracy_m)


      precisions_mac_um, recall_mac_um, f1_score_mac_um, _ = precision_recall_fscore_support(gd_um, pd_um, average='macro')
      precisions_mic_um, recall_mic_um, f1_score_mic_um, _ = precision_recall_fscore_support(gd_um, pd_um, average='micro')
      precisions_wei_um, recall_wei_um, f1_score_wei_um, _ = precision_recall_fscore_support(gd_um, pd_um, average='weighted')
      mac_prec_um.append(precisions_mac_um)
      mac_rec_um.append(recall_mac_um)
      mac_f1_um.append(f1_score_mac_um)
      mic_prec_um.append(precisions_mic_um)
      mic_rec_um.append(recall_mic_um)
      mic_f1_um.append(f1_score_mic_um)
      wei_prec_um.append(precisions_wei_um)
      wei_rec_um.append(recall_wei_um)
      wei_f1_um.append(f1_score_wei_um)
      accuracy_um = accuracy_score(gd_um, pd_um)
      accu_um.append(accuracy_um)


  #####################################################################

    # gold = gold_data[i]
    # pred = predictions[i]
    total_gold += len(gold)
    total_predicted += len(pred)
    
    for a0 in gold:
      for a1 in pred:
        if a0[0] == a1[0] and a0[1] == a1[1]:
          total_unlabeled_matched += 1
          label_confusions.update([(a0[2], a1[2]),])
          if a0[2] == a1[2]:
            total_matched += 1

  prec, recall, f1 = _print_f1(total_gold, total_predicted, total_matched, task_name)
  ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_predicted, total_unlabeled_matched, "Unlabeled " + task_name)
  
  
  
  ######################################################################
  ## my code
  # gd_um = [1]*len(gold)
  # pd_um = []
  # gd_m = [1]*len(gold)
  # pd_m = []
  # for id_new in range(len(gold)):
  #   a0 = gold[id_new]
  #   a1 = pred[id_new]
  #   if a0[0] == a1[0] and a0[1] == a1[1]:
  #     pd_um.append(1)
  #     if a0[2] == a1[2]:
  #       pd_m.append(1)
  #     else:
  #       pd_m.append(0)
  #   else:
  #     pd_um.append(0)


  # precisions_mac_m, recall_mac_m, f1_score_mac_m, _ = precision_recall_fscore_support(gd_m, pd_m, average='macro')
  # precisions_mic_m, recall_mic_m, f1_score_mic_m, _ = precision_recall_fscore_support(gd_m, pd_m, average='micro')
  # precisions_wei_m, recall_wei_m, f1_score_wei_m, _ = precision_recall_fscore_support(gd_m, pd_m, average='weighted')
  # mac_prec_m.append(precisions_mac_m)
  # mac_rec_m.append(recall_mac_m)
  # mac_f1_m.append(f1_score_mac_m)
  # mic_prec_m.append(precisions_mic_m)
  # mic_rec_m.append(recall_mic_m)
  # mic_f1_m.append(f1_score_mic_m)
  # wei_prec_m.append(precisions_wei_m)
  # wei_rec_m.append(recall_wei_m)
  # wei_f1_m.append(f1_score_wei_m)

  print('MATCHED')
  print('mac scores')
  score_avg_prec_m_mac = sum(mac_prec_m) / len(mac_prec_m)
  print('score_avg_prec_m_mac', score_avg_prec_m_mac)
  
  score_avg_rec_m_mac = sum(mac_rec_m) / len(mac_rec_m)
  print('score_avg_recall_m_mac', score_avg_rec_m_mac)
  
  score_avg_f1_m_mac = sum(mac_f1_m) / len(mac_f1_m)
  print('score_avg_f1_m_mac', score_avg_f1_m_mac)

  print('mic scores')
  score_avg_prec_m_mic = sum(mic_prec_m) / len(mic_prec_m)
  print('score_avg_prec_m_mic', score_avg_prec_m_mic)
  
  score_avg_rec_m_mic = sum(mic_rec_m) / len(mic_rec_m)
  print('score_avg_recall_m_mic', score_avg_rec_m_mic)
  
  score_avg_f1_m_mic = sum(mic_f1_m) / len(mic_f1_m)
  print('score_avg_f1_m_mic', score_avg_f1_m_mic)


  print('wei scores')
  score_avg_prec_m_wei = sum(wei_prec_m) / len(wei_prec_m)
  print('score_avg_prec_m_wei', score_avg_prec_m_wei)
  
  score_avg_rec_m_wei = sum(wei_rec_m) / len(wei_rec_m)
  print('score_avg_recall_m_wei', score_avg_rec_m_wei)
  
  score_avg_f1_m_wei = sum(wei_f1_m) / len(wei_f1_m)
  print('score_avg_f1_m_wei', score_avg_f1_m_wei)


  score_avg_accu_m = sum(accu_m) / len(accu_m)
  print('score_avg_accu_m', score_avg_accu_m)


  # precisions_mac_um, recall_mac_um, f1_score_mac_um, _ = precision_recall_fscore_support(gd_um, pd_um, average='macro')
  # precisions_mic_um, recall_mic_um, f1_score_mic_um, _ = precision_recall_fscore_support(gd_um, pd_um, average='micro')
  # precisions_wei_um, recall_wei_um, f1_score_wei_um, _ = precision_recall_fscore_support(gd_um, pd_um, average='weighted')
  # mac_prec_um.append(precisions_mac_um)
  # mac_rec_um.append(recall_mac_um)
  # mac_f1_um.append(f1_score_mac_um)
  # mic_prec_um.append(precisions_mic_um)
  # mic_rec_um.append(recall_mic_um)
  # mic_f1_um.append(f1_score_mic_um)
  # wei_prec_um.append(precisions_wei_um)
  # wei_rec_um.append(recall_wei_um)
  # wei_f1_um.append(f1_score_wei_um)

  print('UNMATCHED')
  print('mac scores')
  score_avg_prec_um_mac = sum(mac_prec_um) / len(mac_prec_um)
  print('score_avg_prec_um_mac', score_avg_prec_um_mac)
  
  score_avg_rec_um_mac = sum(mac_rec_um) / len(mac_rec_um)
  print('score_avg_recall_um_mac', score_avg_rec_um_mac)
  
  score_avg_f1_um_mac = sum(mac_f1_um) / len(mac_f1_um)
  print('score_avg_f1_um_mac', score_avg_f1_um_mac)

  print('mic scores')
  score_avg_prec_um_mic = sum(mic_prec_um) / len(mic_prec_um)
  print('score_avg_prec_um_mic', score_avg_prec_um_mic)
  
  score_avg_rec_um_mic = sum(mic_rec_um) / len(mic_rec_um)
  print('score_avg_recall_um_mic', score_avg_rec_um_mic)
  
  score_avg_f1_um_mic = sum(mic_f1_um) / len(mic_f1_um)
  print('score_avg_f1_um_mic', score_avg_f1_um_mic)


  print('wei scores')
  score_avg_prec_um_wei = sum(wei_prec_um) / len(wei_prec_um)
  print('score_avg_prec_um_wei', score_avg_prec_um_wei)
  
  score_avg_rec_um_wei = sum(wei_rec_um) / len(wei_rec_um)
  print('score_avg_recall_um_wei', score_avg_rec_um_wei)
  
  score_avg_f1_um_wei = sum(wei_f1_um) / len(wei_f1_um)
  print('score_avg_f1_um_wei', score_avg_f1_um_wei)

  score_avg_accu_um = sum(accu_um) / len(accu_um)
  print('score_avg_accu_um', score_avg_accu_um)



  # print('START')
  # print('mac scores')
  # score_avg_prec_st_mac = sum(mac_prec_st) / len(mac_prec_st)
  # print('score_avg_prec_st_mac', score_avg_prec_st_mac)
  
  # score_avg_rec_st_mac = sum(mac_rec_st) / len(mac_rec_st)
  # print('score_avg_recall_st_mac', score_avg_rec_st_mac)
  
  # score_avg_f1_st_mac = sum(mac_f1_st) / len(mac_f1_st)
  # print('score_avg_f1_st_mac', score_avg_f1_st_mac)

  # print('mic scores')
  # score_avg_prec_st_mic = sum(mic_prec_st) / len(mic_prec_st)
  # print('score_avg_prec_st_mic', score_avg_prec_st_mic)
  
  # score_avg_rec_st_mic = sum(mic_rec_st) / len(mic_rec_st)
  # print('score_avg_recall_st_mic', score_avg_rec_st_mic)
  
  # score_avg_f1_st_mic = sum(mic_f1_st) / len(mic_f1_st)
  # print('score_avg_f1_st_mic', score_avg_f1_st_mic)


  # print('wei scores')
  # score_avg_prec_st_wei = sum(wei_prec_st) / len(wei_prec_st)
  # print('score_avg_prec_st_wei', score_avg_prec_st_wei)
  
  # score_avg_rec_st_wei = sum(wei_rec_st) / len(wei_rec_st)
  # print('score_avg_recall_st_wei', score_avg_rec_st_wei)
  
  # score_avg_f1_st_wei = sum(wei_f1_st) / len(wei_f1_st)
  # print('score_avg_f1_st_wei', score_avg_f1_st_wei)



  # print('END')
  # print('mac scores')
  # score_avg_prec_ed_mac = sum(mac_prec_ed) / len(mac_prec_ed)
  # print('score_avg_prec_ed_mac', score_avg_prec_ed_mac)
  
  # score_avg_rec_ed_mac = sum(mac_rec_ed) / len(mac_rec_ed)
  # print('score_avg_recall_ed_mac', score_avg_rec_ed_mac)
  
  # score_avg_f1_ed_mac = sum(mac_f1_ed) / len(mac_f1_ed)
  # print('score_avg_f1_ed_mac', score_avg_f1_ed_mac)

  # print('mic scores')
  # score_avg_prec_ed_mic = sum(mic_prec_ed) / len(mic_prec_ed)
  # print('score_avg_prec_ed_mic', score_avg_prec_ed_mic)
  
  # score_avg_rec_ed_mic = sum(mic_rec_ed) / len(mic_rec_ed)
  # print('score_avg_recall_ed_mic', score_avg_rec_ed_mic)
  
  # score_avg_f1_ed_mic = sum(mic_f1_ed) / len(mic_f1_ed)
  # print('score_avg_f1_ed_mic', score_avg_f1_ed_mic)


  # print('wei scores')
  # score_avg_prec_ed_wei = sum(wei_prec_ed) / len(wei_prec_ed)
  # print('score_avg_prec_ed_wei', score_avg_prec_ed_wei)
  
  # score_avg_rec_ed_wei = sum(wei_rec_ed) / len(wei_rec_ed)
  # print('score_avg_recall_ed_wei', score_avg_rec_ed_wei)
  
  # score_avg_f1_ed_wei = sum(wei_f1_ed) / len(wei_f1_ed)
  # print('score_avg_f1_ed_wei', score_avg_f1_ed_wei)



  # print('LAB')
  # print('mac scores')
  # score_avg_prec_lab_mac = sum(mac_prec_lab) / len(mac_prec_lab)
  # print('score_avg_prec_lab_mac', score_avg_prec_lab_mac)
  
  # score_avg_rec_lab_mac = sum(mac_rec_lab) / len(mac_rec_lab)
  # print('score_avg_recall_lab_mac', score_avg_rec_lab_mac)
  
  # score_avg_f1_lab_mac = sum(mac_f1_lab) / len(mac_f1_lab)
  # print('score_avg_f1_lab_mac', score_avg_f1_lab_mac)

  # print('mic scores')
  # score_avg_prec_lab_mic = sum(mic_prec_lab) / len(mic_prec_lab)
  # print('score_avg_prec_lab_mic', score_avg_prec_lab_mic)
  
  # score_avg_rec_lab_mic = sum(mic_rec_lab) / len(mic_rec_lab)
  # print('score_avg_recall_lab_mic', score_avg_rec_lab_mic)
  
  # score_avg_f1_lab_mic = sum(mic_f1_lab) / len(mic_f1_lab)
  # print('score_avg_f1_lab_mic', score_avg_f1_lab_mic)


  # print('wei scores')
  # score_avg_prec_lab_wei = sum(wei_prec_lab) / len(wei_prec_lab)
  # print('score_avg_prec_lab_wei', score_avg_prec_lab_wei)
  
  # score_avg_rec_lab_wei = sum(wei_rec_lab) / len(wei_rec_lab)
  # print('score_avg_recall_lab_wei', score_avg_rec_lab_wei)
  
  # score_avg_f1_lab_wei = sum(wei_f1_lab) / len(wei_f1_lab)
  # print('score_avg_f1_lab_wei', score_avg_f1_lab_wei)
  
  
  return prec, recall, f1, ul_prec, ul_recall, ul_f1, label_confusions




def compute_unlabeled_span_f1(gold_data, predictions, task_name):
  assert len(gold_data) == len(predictions)
  total_gold = 0
  total_predicted = 0
  total_matched = 0
  total_unlabeled_matched = 0
  label_confusions = Counter()  # Counter of (gold, pred) label pairs.

  for i in range(len(gold_data)):
    gold = gold_data[i]
    pred = predictions[i]
    total_gold += len(gold)
    total_predicted += len(pred)
    for a0 in gold:
      for a1 in pred:
        if a0[0] == a1[0] and a0[1] == a1[1]:
          total_unlabeled_matched += 1
          label_confusions.update([(a0[2], a1[2]),])
          if a0[2] == a1[2]:
            total_matched += 1
  prec, recall, f1 = _print_f1(total_gold, total_predicted, total_matched, task_name)
  ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_predicted, total_unlabeled_matched, "Unlabeled " + task_name)
  return prec, recall, f1, ul_prec, ul_recall, ul_f1, label_confusions


def compute_relation_f1(sentences, gold_rels, predictions):
  assert len(gold_rels) == len(predictions)
  total_gold = 0
  total_predicted = 0
  total_matched = 0
  total_unlabeled_matched = 0
  label_confusions = Counter()
  # Compute unofficial F1 of entity relations.
  doc_id = 0  # Actually sentence id.
  gold_tuples = []  # For official eval.
  predicted_tuples = []
  for gold, prediction in zip(gold_rels, predictions):
    total_gold += len(gold)
    total_predicted += len(prediction)
    # print " ".join(sentences[doc_id])
    # print "Gold:", gold
    # print "Prediction:", prediction
    for g in gold:
      gold_tuples.append([["d{}_{}_{}".format(doc_id, g[0], g[1]),
                           "d{}_{}_{}".format(doc_id, g[2], g[3])], g[4]])
      for p in prediction:
        if g[0] == p[0] and g[1] == p[1] and g[2] == p[2] and g[3] == p[3]:
          total_unlabeled_matched += 1
          if g[4] == p[4]:
            total_matched += 1
          break
    for p in prediction:
      predicted_tuples.append([["d{}_{}_{}".format(doc_id, p[0], p[1]),
                                "d{}_{}_{}".format(doc_id, p[2], p[3])], p[4]])
    doc_id += 1
  precision, recall, f1 = _print_f1(total_gold, total_predicted, total_matched, "Relations (unofficial)")
  ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_predicted, total_unlabeled_matched, "Unlabeled (unofficial)")
  relation_metrics.span_metric(gold_tuples, predicted_tuples)
  print('REL gold_tuples',gold_tuples)
  print('REL predicted_tuples',predicted_tuples)
  return precision, recall, f1
 

def compute_srl_f1(sentences, gold_srl, predictions, srl_conll_eval_path):
  assert len(gold_srl) == len(predictions)
  total_gold = 0
  total_predicted = 0
  total_matched = 0
  total_unlabeled_matched = 0
  comp_sents = 0
  label_confusions = Counter()

  # Compute unofficial F1 of SRL relations.
  for gold, prediction in zip(gold_srl, predictions):
    gold_rels = 0
    pred_rels = 0
    matched = 0
    for pred_id, gold_args in gold.iteritems():
      filtered_gold_args = [a for a in gold_args if a[2] not in ["V", "C-V"]]
      total_gold += len(filtered_gold_args)
      gold_rels += len(filtered_gold_args)
      if pred_id not in prediction:
        continue
      for a0 in filtered_gold_args:
        for a1 in prediction[pred_id]:
          if a0[0] == a1[0] and a0[1] == a1[1]:
            total_unlabeled_matched += 1
            label_confusions.update([(a0[2], a1[2]),])
            if a0[2] == a1[2]:
              total_matched += 1
              matched += 1
    for pred_id, args in prediction.iteritems():
      filtered_args = [a for a in args if a[2] not in ["V"]] # "C-V"]] 
      total_predicted += len(filtered_args)
      pred_rels += len(filtered_args)
  
    if gold_rels == matched and pred_rels == matched:
      comp_sents += 1

  precision, recall, f1 = _print_f1(total_gold, total_predicted, total_matched, "SRL (unofficial)")
  ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_predicted, total_unlabeled_matched, "Unlabeled SRL (unofficial)")

  # Prepare to compute official F1.
  if not srl_conll_eval_path:
    print "No gold conll_eval data provided. Recreating ..."
    gold_path = "/tmp/srl_pred_%d.gold" % os.getpid()
    print_to_conll(sentences, gold_srl, gold_path, None)
    gold_predicates = None
  else:
    gold_path = srl_conll_eval_path
    gold_predicates = read_gold_predicates(gold_path)

  temp_output = "/tmp/srl_pred_%d.tmp" % os.getpid()
  print_to_conll(sentences, predictions, temp_output, gold_predicates)

  # Evalute twice with official script.
  child = subprocess.Popen('sh {} {} {}'.format(
      _SRL_CONLL_EVAL_SCRIPT, gold_path, temp_output), shell=True, stdout=subprocess.PIPE)
  eval_info = child.communicate()[0]
  child2 = subprocess.Popen('sh {} {} {}'.format(
      _SRL_CONLL_EVAL_SCRIPT, temp_output, gold_path), shell=True, stdout=subprocess.PIPE)
  eval_info2 = child2.communicate()[0]  
  try:
    conll_recall = float(eval_info.strip().split("\n")[6].strip().split()[5])
    conll_precision = float(eval_info2.strip().split("\n")[6].strip().split()[5])
    if conll_recall + conll_precision > 0:
      conll_f1 = 2 * conll_recall * conll_precision / (conll_recall + conll_precision)
    else:
      conll_f1 = 0
    print(eval_info)
    print(eval_info2)
    print("Official CoNLL Precision={}, Recall={}, Fscore={}".format(
        conll_precision, conll_recall, conll_f1))
  except IndexError:
    conll_recall = 0
    conll_precision = 0
    conll_f1 = 0
    print("Unable to get FScore. Skipping.") 

  return precision, recall, f1, conll_precision, conll_recall, conll_f1, ul_prec, ul_recall, ul_f1, label_confusions, comp_sents
  

def print_sentence_to_conll(fout, tokens, labels):
  """Print a labeled sentence into CoNLL format.
  """
  for label_column in labels:
    assert len(label_column) == len(tokens)
  for i in range(len(tokens)):
    fout.write(tokens[i].ljust(15))
    for label_column in labels:
      fout.write(label_column[i].rjust(15))
    fout.write("\n")
  fout.write("\n")


def read_gold_predicates(gold_path):
  fin = codecs.open(gold_path, "r", "utf-8")
  gold_predicates = [[],]
  for line in fin:
    line = line.strip()
    if not line:
      gold_predicates.append([])
    else:
      info = line.split()
      gold_predicates[-1].append(info[0])
  fin.close()
  return gold_predicates


def print_to_conll(sentences, srl_labels, output_filename, gold_predicates):
  fout = codecs.open(output_filename, "w", "utf-8")
  for sent_id, words in enumerate(sentences):
    if gold_predicates:
      assert len(gold_predicates[sent_id]) == len(words)

    pred_to_args = srl_labels[sent_id]
    props = ["-" for _ in words]
    col_labels = [["*" for _ in words] for _ in range(len(pred_to_args))]

    for i, pred_id in enumerate(sorted(pred_to_args.keys())):
      # To make sure CoNLL-eval script count matching predicates as correct.
      if gold_predicates and gold_predicates[sent_id][pred_id] != "-":
        props[pred_id] = gold_predicates[sent_id][pred_id]
      else:
        props[pred_id] = "P" + words[pred_id]
      flags = [False for _ in words]
      for start, end, label in pred_to_args[pred_id]:
        # Unfortunately, gold CoNLL-2012 data has overlapping args.
        if not max(flags[start:end+1]):
          col_labels[i][start] = "(" + label + col_labels[i][start]
          col_labels[i][end] = col_labels[i][end] + ")"
          for j in range(start, end+1):
            flags[j] = True
      # Add unpredicted verb (for predicted SRL).
      if not flags[pred_id]:
        col_labels[i][pred_id] = "(V*)"
    print_sentence_to_conll(fout, props, col_labels)

  fout.close()


def print_to_iob2(sentences, gold_ner, pred_ner, gold_file_path):
  """Print to IOB2 format for NER eval. 
  """
  # Write NER prediction to IOB format.
  temp_file_path = "/tmp/ner_pred_%d.tmp" % os.getpid()
  # Read IOB tags from preprocessed gold path.
  gold_info = [[]]
  if gold_file_path:
    fgold = codecs.open(gold_file_path, "r", "utf-8")
    for line in fgold:
      line = line.strip()
      if not line:
        gold_info.append([])
      else:
        gold_info[-1].append(line.split())
  else:
    fgold = None
  fout = codecs.open(temp_file_path, "w", "utf-8")
  for sent_id, words in enumerate(sentences):
    pred_tags = ["O" for _ in words]
    for start, end, label in pred_ner[sent_id]:
      pred_tags[start] = "B-" + label
      for k in range(start + 1, end + 1):
        pred_tags[k] = "I-" + label
    if not fgold:
      gold_tags = ["O" for _ in words]
      for start, end, label in gold_ner[sent_id]:
        gold_tags[start] = "B-" + label
        for k in range(start + 1, end + 1):
          gold_tags[k] = "I-" + label
    else:
      assert len(gold_info[sent_id]) == len(words)
      gold_tags = [t[1] for t in gold_info[sent_id]] 
    for w, gt, pt in zip(words, gold_tags, pred_tags):
      fout.write(w + " " + gt + " " + pt + "\n")
    fout.write("\n")
  fout.close()
  child = subprocess.Popen('./ner/bin/conlleval < {}'.format(temp_file_path),
                           shell=True, stdout=subprocess.PIPE)
  eval_info = child.communicate()[0]
  print eval_info

