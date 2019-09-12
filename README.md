# MS-intern-project

This project aims to increase the usage of purchased workloads for O365 SMB using a recommender system. To achieve this goal, our model will 1) predict the usage pattern for a tenant based on the profile and purchase information; 2) make recommendations based on the gap between predicted usage and actual usage, which indicates the potential for usage growth. We build a deep neural network based usage prediction model and showed that it outperforms traditional regression models.

This repo contains the code for data preprocessing, building and training multi-task neural networks and hypeprparameter tuning using random search. 


# Dependencies

- Python 3.5
- TensorFlow >= 1.7
- Other required packages are summarized in `requirements.txt`.

# Quickstart

After installing the dependencies, run the following command to replicate the experiment results.

```
./run_orig_model_clipdata_tryFasterTuning.sh
```

Then you can use `load-NN-best-with-clip-data.ipynb` to load and analyze the best model.

# Data encoding

`encode_data.py` can be used to encode a csv file into numpy arrays. It takes two files as inputs: the raw data file (CSV file) and a corresponding configuration file (a metadata JSON file that describes the datatype of each column). `raw_data/create-configure-metadata.ipynb` is an example that generates a configuration file.

`encode_data.py` performs four steps:
- Split dataset into training, dev and test set.
- Clarify the input features and output labels.
- Encode different data type inputs.
- Concatenate and normalize all the encoded sub-inputs.
