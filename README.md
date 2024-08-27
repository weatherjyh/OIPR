# OIPR

Repository  for the Operator Interest-based Precision and Recall metrics.

Using the following codes to calculated the OIPR metrics:

```
from evaluation_methods import evaluate_OIPR
OIPR_config = {'l_dis': 5, 'l_obs': 20, 'b_dur': 0.5}
precision, recall, f1_score = evaluate_OIPR(gt, pred, **OIPR_config)
```

