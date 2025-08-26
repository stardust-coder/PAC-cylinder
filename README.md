# Phase amplitude coupling with cylindrical model

## Files
- model/
- experiment.py ... main code

## Usage

### MLE on ECoG dataset
Comment out each paragraph you want to use before running the script.

[Lists of Paragraph]
1. Load Marmoset ECoG with prerpocessing and visualize pairplot
    - line148~
2. MLE of Gamma-von Mises model
    - Single MLE
        - line159~
    - repeating with 1Hz intervals and plot comodulogram of MI and Kappa_MLE. (not recommended)
        - line174~line248
3. Estimation of Generalized Gamma-von Mises model with EM algorithm
    - line249~

Then run
```bash
python experiment.py
```

### Score matching on simulation dataset
The ground truth model is Gamma von Mises. Run to compare score matching and MLE.
```bash
python score_matching.py
```


## Marmoset Dataset
We set the [Marmoset Auditory Dataset 01 (DataID: 4924)](https://dataportal.brainminds.jp/ecog-auditory-01) in `../data/`.
```
Komatsu, M; Ichinohe N (2020): Effects of ketamine administration on auditory information processing in the neocortex of nonhuman primates. Front. Psychiatry 11:826. doi: 10.3389/fpsyt.2020.00826
```

## How to cite
```
Coming soon...
```