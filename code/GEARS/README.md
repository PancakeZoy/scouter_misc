To train `GEARS` on all datasets, please run the following:

```
for i in {1..10}; do python run_gears_dixit.py $i; done
for i in {1..5}; do python run_gears_adamson.py $i; done
for i in {1..5}; do python run_gears_norman.py $i; done
for i in {1..5}; do python run_gears_k562.py $i; done
for i in {1..5}; do python run_gears_rpe1.py $i; done
```