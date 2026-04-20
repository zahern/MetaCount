Book manual model comparison (from fitted specifications)

Fit metrics

| Rank | Model | BIC | AIC | Log-Likelihood | Parameters | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | NB latent-class (book) | -39727343377324.86 | -39727343377368.26 | 19863671688696.13 | 12 | book latent-class spec; coef=ok |
| 2 | NB baseline (book) | 1925.5156349389833 | 1900.1982372553173 | -943.0991186276586 | 7 | book baseline spec; coef=ok |

Key structural differences

| Section | Difference |
| --- | --- |
| fixed_terms | NB only: CURVES, LENGTH |
| rdm_terms | LC only: CURVES:lognormal, LENGTH:normal |
| membership_terms | LC only: FC_ENCODED |
| latent_classes | NB=1 -> LC=2 |