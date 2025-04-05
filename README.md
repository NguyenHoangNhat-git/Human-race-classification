# Human-race-classification

A simple human(male) race classification model(slightly smaller Resnet18) using a portion of the FairFace dataset (~45K imgs: both train and val)

Note:
- This is my first "complete" ML project with many of the ML procedures that i rebuilt from scratch, so most of everything was experimental
- The Fairface dataset was orginally trained on resnet50(containing ~ 25millions parameters), which is ~2000 times more parameters more than my model. In my problem, i used less images and simpler model to fit a more specific purpose, leading to faster computing time but lower performance
- Best: 59% val accuracy 
- Due to lack of computing power, I used 1 setup(architecture, hyperparameters) to train several models for each age group and apply a similar technique to bagging(ensemble) for them to vote for predictions

## Usage

Run the script with:

```bash
python3 predict.py sample.jpg
```
