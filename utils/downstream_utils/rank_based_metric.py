from sklearn.metrics import auc

def line_level_evaluation(all_lines_score: list, flaw_line_indices: list, top_k_loc: list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], top_k_constant: list = [10],
                          true_positive_only: bool = True, index=None):
    if true_positive_only:
        # line indices ranking based on attr values
        ranking = sorted(range(len(all_lines_score)), key=lambda i: all_lines_score[i], reverse=True)
        # total flaw lines
        num_of_flaw_lines = len(flaw_line_indices)
        # clean lines + flaw lines
        total_lines = len(all_lines_score)
        ### TopK% Recall ###
        all_correctly_predicted_flaw_lines = []
        ### IFA ###
        ifa = True
        all_clean_lines_inspected = []
        for top_k in top_k_loc:
            correctly_predicted_flaw_lines = 0
            for indice in flaw_line_indices:
                # if within top-k
                k = int(len(all_lines_score) * top_k)
                # if detecting any flaw lines
                if indice in ranking[: k]:
                    correctly_predicted_flaw_lines += 1
                if ifa:
                    # calculate Initial False Alarm
                    # IFA counts how many clean lines are inspected until the first vulnerable line is found when inspecting the lines ranked by the approaches.
                    flaw_line_idx_in_ranking = ranking.index(indice)
                    # e.g. flaw_line_idx_in_ranking = 3 will include 1 vulnerable line and 3 clean lines
                    all_clean_lines_inspected.append(flaw_line_idx_in_ranking)
                    # for IFA
            min_clean_lines_inspected = min(all_clean_lines_inspected)
            # for All Effort
            max_clean_lines_inspected = max(all_clean_lines_inspected)
            # only do IFA and All Effort once
            ifa = False
            # append result for one top-k value
            all_correctly_predicted_flaw_lines.append(correctly_predicted_flaw_lines)

        ### Top10 Accuracy ###
        all_correctly_localized_func = []
        top_10_correct_idx = []
        top_10_not_correct_idx = []
        correctly_located = False
        for k in top_k_constant:
            for indice in flaw_line_indices:
                # if detecting any flaw lines
                if indice in ranking[: k]:
                    """
                    # extract example for the paper
                    if index == 2797:
                        print("2797")
                        print("ground truth flaw line index: ", indice)
                        print("ranked line")
                        print(ranking)
                        print("original score")
                        print(all_lines_score)
                    """
                    # append result for one top-k value
                    all_correctly_localized_func.append(1)
                    correctly_located = True
                else:
                    all_correctly_localized_func.append(0)
            if correctly_located:
                top_10_correct_idx.append(index)
            else:
                top_10_not_correct_idx.append(index)
        results = {"total_lines": total_lines,
                    "num_of_flaw_lines": num_of_flaw_lines,
                    "all_correctly_predicted_flaw_lines": all_correctly_predicted_flaw_lines,
                    "all_correctly_localized_function": all_correctly_localized_func,
                    "min_clean_lines_inspected": min_clean_lines_inspected,
                    "max_clean_lines_inspected": max_clean_lines_inspected,
                    "top_10_correct_idx": top_10_correct_idx,
                    "top_10_not_correct_idx": top_10_not_correct_idx}
        return results
        # return total_lines, num_of_flaw_lines, all_correctly_predicted_flaw_lines, min_clean_lines_inspected, max_clean_lines_inspected, all_correctly_localized_func, \
        #        top_10_correct_idx, top_10_not_correct_idx
    else:
        # all_lines_score_with_label: [[line score, line level label], [line score, line level label], ...]
        all_lines_score_with_label = []
        for i in range(len(all_lines_score)):
            if i in flaw_line_indices:
                all_lines_score_with_label.append([all_lines_score[i], 1])
            else:
                all_lines_score_with_label.append([all_lines_score[i], 0])
        return all_lines_score_with_label

class LineVulMetric:
    top_k_locs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    top_k_constant = [10]

    def __init__(self):
        self.sum_total_lines = 0
        self.total_flaw_lines = 0
        self.total_function = 0
        self.total_min_clean_lines_inspected = 0
        self.total_max_clean_lines_inspected = 0
        self.total_correctly_predicted_flaw_lines = [0 for _ in range(len(self.top_k_locs))]
        self.total_correctly_localized_function = [0 for _ in range(len(self.top_k_constant))]

    def score(self, line_scores, flaw_line_indices):
        line_eval_results = line_level_evaluation(line_scores, flaw_line_indices)

        self.total_function += 1
        self.sum_total_lines += line_eval_results["total_lines"]
        self.total_flaw_lines += line_eval_results["num_of_flaw_lines"]
        # IFA metric
        self.total_min_clean_lines_inspected += line_eval_results["min_clean_lines_inspected"]

        # # For IFA Boxplot
        # ifa_records.append(line_eval_results["min_clean_lines_inspected"])

        # For Top-10 Acc Boxplot
        # todo
        # top_10_acc_records.append(line_eval_results[])

        # All effort metric
        self.total_max_clean_lines_inspected += line_eval_results["max_clean_lines_inspected"]
        for j in range(len(self.top_k_locs)):
            self.total_correctly_predicted_flaw_lines[j] += line_eval_results["all_correctly_predicted_flaw_lines"][j]
        # top 10 accuracy
        for k in range(len(self.top_k_constant)):
            self.total_correctly_localized_function[k] += line_eval_results["all_correctly_localized_function"][k]

        # top 10 correct idx and not correct idx
        # if line_eval_results["top_10_correct_idx"] != []:
        #     self.all_top_10_correct_idx.append(line_eval_results["top_10_correct_idx"][0])
        # if line_eval_results["top_10_not_correct_idx"] != []:
        #     self.all_top_10_not_correct_idx.append(line_eval_results["top_10_not_correct_idx"][0])

    def get_metric(self):
        top_20_recall = [round(self.total_correctly_predicted_flaw_lines[i] / self.total_flaw_lines, 2) * 100 for i in range(len(self.top_k_locs))]
        top_10_acc = [round(self.total_correctly_localized_function[i] / self.total_function, 2) * 100 for i in range(len(self.top_k_constant))]
        ifa = round(self.total_min_clean_lines_inspected / self.total_function, 2)
        recall_topk_loc_auc = auc(x=self.top_k_locs, y=[round(self.total_correctly_predicted_flaw_lines[i] / self.total_flaw_lines, 2) for i in range(len(self.top_k_locs))])
        total_effort = round(self.total_max_clean_lines_inspected / self.sum_total_lines, 2)
        avg_line_in_one_func = self.sum_total_lines / self.total_function
        total_function = self.total_function

        return {
            "top_20%_recall": top_20_recall,
            "top_10_acc": top_10_acc,
            "ifa": ifa,
            "recall@topk%loc_auc": recall_topk_loc_auc,
            "total_effort": total_effort,
            "avg_line_in_one_func": avg_line_in_one_func,
            "total_function": total_function
        }