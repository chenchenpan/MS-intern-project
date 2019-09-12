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
