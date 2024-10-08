from searchlogit import MixedLogit
import pandas as pd
import debugpy
debugpy.breakpoint()
df = pd.read_csv(
    "https://raw.githubusercontent.com/arteagac/xlogit/master/examples/data/electricity_long.csv")

# reverse sign for randvars with a lognormal distribution
df['tod'] = -df['tod']
df['seas'] = -df['seas']
varnames = ['pf', 'cl', 'loc', 'wk', 'tod', 'seas']
X = df[varnames]
y = df['choice']

# Fit the model with xlogit
model = MixedLogit()
model.fit(X, y,
          varnames,
          alts=df['alt'],
          ids=df['chid'],
          panels=df['id'],
          randvars={'cl': 'n', 'loc': 'n',
                    'wk': 'u', 'tod': 'ln', 'seas': 'n'},
          correlation=['loc', 'wk', 'tod', 'seas'],
          transvars=['cl'],
          n_draws=600
          )
model.summary()
