# model/

This folder holds the trained model that the API loads at startup:

- `churn_pipeline.joblib`: preprocessing + XGBoost in a single file
- `metadata.json`: decision threshold, test metrics, feature list and library versions

Create or refresh both files from the repository root with:

```bash
python train_model.py
```

The Docker build runs this command automatically, so the deployed container always has a model that matches the installed libraries.
