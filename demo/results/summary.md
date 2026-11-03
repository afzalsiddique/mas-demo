# Demo results

`{"k": 4, "repeats": 5, "n_questions": 5, "model": "qwen2.5-7b-instruct", "temperature": 0.7, "max_tokens_per_call": 768, "mean_tokens_team_run": 5973, "mean_tokens_single_sample": 1684}`

## By condition

| condition   | accuracy_em   | acc_sd_over_repeats   |   f1 |   calls_per_q |   tokens_per_q |   tok_sd_over_repeats |   errors |
|:------------|:--------------|:----------------------|-----:|--------------:|---------------:|----------------------:|---------:|
| A_single    | 68%           | 11 pts                | 0.68 |           2   |           1684 |                   118 |        0 |
| B_vote@4    | 72%           | 11 pts                | 0.72 |           8   |           7051 |                   615 |        0 |
| C_team      | 72%           | 11 pts                | 0.79 |           9.2 |           5973 |                   481 |        0 |

## Accuracy per question (mean EM over repeats)

|                                            |   A_single |   B_vote@4 |   C_team |
|:-------------------------------------------|-----------:|-----------:|---------:|
| ('5a7574e25542996c70cfaf01', 'comparison') |        1   |        1   |      1   |
| ('5a77bdcb5542997042120b0b', 'comparison') |        1   |        1   |      1   |
| ('5ab43b8a5542991779162c22', 'bridge')     |        0.8 |        1   |      1   |
| ('5ae56a3f5542992663a4f1f4', 'bridge')     |        0   |        0   |      0.6 |
| ('5ae5da8d55429929b08079d1', 'bridge')     |        0.6 |        0.6 |      0   |
