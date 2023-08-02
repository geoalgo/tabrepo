# Notes regarding the stacking simulation 
* We need to add an internal function to create the monotonic constraints as the feature order is switched internally.
  * either we activate monotonic for all of l2 and we write function to solve this or we write a map from outer feature data to inner feature data 
* Technically, we need to change the random seed for the simulation as we now have the wrong (i.e., the same) split as in L1
* Moreover, if we add such a flag, then we also need to change the naming to L2.
* We need to disable preprocessing for autogluon and add feature metadata obtained from preprocessing by us