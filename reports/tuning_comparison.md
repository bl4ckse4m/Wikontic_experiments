# Comparison (proxy + gold/silver)

annotation: 0 human gold, 37 LLM-assisted silver (silver != gold)

| method | ok_rate | n_facts | sem.facts | lex.support | compl.tok | facts/1kchars | gold P/R/F1 core | gold P/R/F1 full |
|---|---|---|---|---|---|---|---|---|
| coverage | 0.998 | 19476 | 19384 | 0.852 | 5196190 | 5.35 | 0.378/0.302/0.336 | 0.267/0.213/0.237 |
| baseline_fixed | 0.999 | 18952 | 18882 | 0.845 | 4395953 | 5.38 | 0.378/0.293/0.330 | 0.261/0.203/0.228 |
| fewshot | 0.997 | 19039 | 18966 | 0.847 | 5440979 | 5.23 | 0.340/0.274/0.304 | 0.228/0.184/0.203 |
| xprompt_union | 1.000 | 40303 | 40279 | 0.824 | — | 5.13 | 0.283/0.498/0.361 | 0.188/0.331/0.240 |
| xprompt_vote2 | 1.000 | 12200 | 12198 | 0.874 | — | 5.18 | 0.445/0.207/0.282 | 0.341/0.158/0.216 |

Overlap: {"n_shared_inputs": 1825, "n_semantic_fact_clusters": 39363, "unique_to_one_method": {}, "present_in_all_methods": 4683, "pairwise_shared_clusters": {"baseline_fixed|coverage": 7445, "baseline_fixed|xprompt_union": 18490, "baseline_fixed|xprompt_vote2": 9366, "coverage|xprompt_union": 19019, "coverage|xprompt_vote2": 10212, "xprompt_union|xprompt_vote2": 12134, "baseline_fixed|fewshot": 6607, "coverage|fewshot": 7449, "fewshot|xprompt_union": 18672, "fewshot|xprompt_vote2": 9371}}