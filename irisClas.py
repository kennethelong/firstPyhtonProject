import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import mglearn
import numpy as np
import time

def generate_observation(mean_values, range_values):
    # Generate each feature's value within its mean ± half of the range
    new_observation = [
        np.random.uniform(mean - (r / 2), mean + (r / 2))
        for mean, r in zip(mean_values, range_values)
    ]

    # Convert to desired format
    return np.array([new_observation])

start_time = time.time()
iris_dataset = load_iris()

iris_df = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)

## Setting up the test and training data sets
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)

## Fitting the training and test data
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)*100))

print("Test set score: {:.2f}".format(knn.score(X_test, y_pred)))


## --------------------------------------------------------------##
end_time = time.time()

print(f"\nExecution time: {end_time - start_time} seconds")





#print("Keys of iris_dataset:\n", iris_dataset.keys())

#print(iris_dataset['DESCR'][:293] + "\n...")
#print("Target names:\n", iris_dataset['target_names'])
# print("\nFeature names:\n", iris_dataset['feature_names'])
# print("\nType of data:\n", type(iris_dataset['data']))
# print("\nShape of data:\n", iris_dataset['data'].shape)
# print("\nFirst five rows of data:\n", iris_dataset['data'][:5])
# print("\nType of target:\n", type(iris_dataset['target']))
# print("\nShape of target:\n", iris_dataset['target'].shape)
# print("\nTarget:\n", iris_dataset['target'])


#
# print("\nX_train shape", X_train.shape)
# print("\ny_train shape", y_train.shape)
#
# print("\nX_test shape", X_test.shape)
# print("\ny_test shape", y_test.shape)

# iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# pd.plotting.scatter_matrix(iris_dataframe, c = y_train, figsize = (15,15), marker = 'o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)
#
# #plt.show()
#

# print(knn)
#
# print("\n", knn.get_params())

# ## Making a new observation
# X_new = np.array([[5,2.9,1,0.2]])
# print("X_new.shape", X_new.shape)
#
# X_new_auto = generate_observation(mean_values, range_values)
# print("Generated observation:", X_new_auto)
#
# ## Predicting which flower the observations is based on the model
# prediction = knn.predict(X_new)
# print("Prediction: ", prediction)
# print("Prediction target name: ", iris_dataset['target_names'][prediction])
#
# prediction = knn.predict(X_new_auto)
# print("Prediction: ", prediction)
# print("Prediction target name: ", iris_dataset['target_names'][prediction])

# # Display the results
# print("Mean values for each feature:", mean_values)
# print("Range values for each feature:", range_values)

# # Calculate mean and range for each feature
# mean_values = iris_df.mean().values
# range_values = (iris_df.max() - iris_df.min()).values