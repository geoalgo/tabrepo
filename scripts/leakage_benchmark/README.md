# Notes regarding the simulation of stacking with AutoGluon 

## Monotonic Constraints 
* Problem: the feature order changes internally, and thus we can not define the constraints at fit time.  
* Solution AutoGluon: We need to add an internal function to create the monotonic constraints as the feature order is switched internally.
* Simulation Workaround: Either we activate monotonic for all stack features or we write a map from outer feature data to inner feature data 

## Random Seed 
* Problem: the seed needs to change per layer to avoid two layers having the same cross-validation splits. 
* Solution: allow the base seed to be passed at fit time and change it accordingly (already possible but not documented)
  * What you have to do: pass `learner_kwargs=dict(random_state=X)` during init of `TabularPredictor`
    * The CV split is based on self._random_state, [here](https://github.com/LennartPurucker/autogluon/blob/1cd5f98db131e3dd8e540365eb7edb2a6669539b/core/src/autogluon/core/models/ensemble/bagged_ensemble_model.py#L171).
    * The random state is set per layer [here](https://github.com/LennartPurucker/autogluon/blob/1cd5f98db131e3dd8e540365eb7edb2a6669539b/core/src/autogluon/core/trainer/abstract_trainer.py#L628).
    * The default init of the random state is [here](https://github.com/LennartPurucker/autogluon/blob/3f48d660111f912509470bfb237070602ad2c5ff/tabular/src/autogluon/tabular/learner/default_learner.py#L32) and the corresponding class is [here](https://github.com/LennartPurucker/autogluon/blob/3f48d660111f912509470bfb237070602ad2c5ff/tabular/src/autogluon/tabular/learner/abstract_learner.py#L57) and finally the value is set [here](https://github.com/LennartPurucker/autogluon/blob/1cd5f98db131e3dd8e540365eb7edb2a6669539b/core/src/autogluon/core/learner/abstract_learner.py#L32). 
    * The default value can be modified [here](https://github.com/LennartPurucker/autogluon/blob/1cd5f98db131e3dd8e540365eb7edb2a6669539b/tabular/src/autogluon/tabular/predictor/predictor.py#L257).

## Preprocessing
* Problem: L1 already preprocessed the data and L2 should not preprocess the data again
* Solution: manually define feature preprocessor, and pass feature metadata from l1 to AutoGluon. 
  * Additionally, the stack features must be in the feature metadata and maybe their special type must be set.

## Model Naming Conventions 
* Problem: By default, the first layer's models end with L2. We need to change this in AutoGluon to start with L2.
* Solution: change layer i
* Simulation Workaround: ignore this and change names post hoc

