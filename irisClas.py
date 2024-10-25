from sklearn.datasets import load_iris
import time

start_time = time.time()
iris_dataset = load_iris()

#print("Keys of iris_dataset:\n", iris_dataset.keys())

#print(iris_dataset['DESCR'][:293] + "\n...")
#print("Target names:\n", iris_dataset['target_names'])
print("\nFeature names:\n", iris_dataset['feature_names'])
print("\nType of data:\n", type(iris_dataset['data']))
print("\nShape of data:\n", iris_dataset['data'].shape)
print("\nFirst five rows of data:\n", iris_dataset['data'][5])

end_time = time.time()

print(f"\nExecution time: {end_time - start_time} seconds")
