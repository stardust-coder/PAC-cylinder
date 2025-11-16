# Phase amplitude coupling with cylindrical model

## Files
- model/
- experiment.py ... main code

## Usage

### 1-1. MLE on bivariate simulation dataset


### 1-2. MLE on trivariate simulation dataset

The simulation is conducted on synthetic time series data generated from pink noises and artificial modulation on PAC and AAC, completely following Nadalin et al.

Ours
```
python trivariate_sim_MLE.py
```

[Nadalin et al., 2019]
```
python -m main.GLMCFC
```

### 1-3. Compare our method with existing methods

`trivariate_sim_comparison.py` compares GLM-CFC, Alowφlow model with the proposed method.

`trivariate_sim_correlation.py` compares classical MVL/correlation with the proposed method.



### 2-1. MLE on real dataset
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
python bivariate_ecog.py
```

### 2-2. MLE on trivariate real dataset

`trivariate_real_ecog.py` loads marmoset ECoG dataset.

`trivariate_real_humanMEA.py` uses Human seizure MEA dataset from Nadalin et al. It compares the proposed method and the classical MVL/correlation.

`trivariate_real_rodentLFP.py` is meant to be same as `trivariate_real_humanMEA.py` but with different dataset.



### 3-1. EM algorithm of MoGGM (Mixture of Generalized Gamma von Mises)
```
python Gam_vM_EM.py
python GGam_v_EM.py
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