# Automatic DNN Builder with Adanet & Tensorflow

## Includes a basic and an advanced model
Basic MSE = 0.054

Advanced MSE = 0.033

## Modifying would be very easy
1. Swap out the method which builds the tensorflow dataset with one that supports your data size
2. modify candidate `LAYER_SIZE`, regressors and response variables as appropriate. *Uses boston housing toy dataset.*
3. familiarize yourself with this https://www.tensorflow.org/guide/feature_columns
4. modify the tf.estimator to the above requirements.
5. Serve models after exporting `estimator.save_model` 
