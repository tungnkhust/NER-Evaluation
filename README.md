# NER-Evaluation

All most metric to evaluate model use typically metrics such as `precision score`, `recall score`, `f1-score` at token level.
But this is not accurate to evaluate the performance of system. So, I build more useful metric to evaluate at a entity level.
This repository, I implement metrics which introduced by [MUC](https://www.aclweb.org/anthology/M93-1007/).
The metrics are define as terms:
- Correct(cor): Both are the same.
- Incorrect(inc): The predicted entity and the true entity don’t match
- Partial(par): Both are the same entity but the boundaries of the surface string wrong
- Missing(mis): The system doesn't predict entity
- Spurius(spu): The system predict entity which doesn't exist in the true label.

And the `precision score`, `recall score`, `f1-score` are computed with different evaluation schema.
```
possible(pos) = cor + inc + par + mis = TP + FN
actual(act) = cor + inc + par + spu = TP + FP
precision_score = cor/act = TP/(TP + FP)
recall_score = cor/pos = TP/(TP + FN)
f1-score = 2*precision_score*recall_score/(precision_score + recall_score)
```