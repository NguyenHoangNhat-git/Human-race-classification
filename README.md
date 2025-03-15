# Human-race-classification

A simple human(male) race classification model(slightly smaller Resnet18) using the FairFace dataset (~45K imgs: both train and val)

Note:

- This is my first (complete) ML project, so most of everything was experimental
- Very low performance, 53% val accuracy on the Fairface dataset
- Due to lack of computing power, I used 1 setup(architecture, hyperparameters) to train several models for each age group and apply a similar technique to bagging(ensemble) for them to vote for predictions

## Usage

Run the script with:

```bash
python3 predict.py sample.jpg
```
