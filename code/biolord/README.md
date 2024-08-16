To train `biolord` on all datasets, please run the following:

```
for i in {1..10}; do python run_biolord_dixit.py $i; done
for i in {1..5}; do python run_biolord_adamson.py $i; done
for i in {1..5}; do python run_biolord_norman.py $i; done
for i in {1..5}; do python run_biolord_k562.py $i; done
for i in {1..5}; do python run_biolord_rpe1.py $i; done
```