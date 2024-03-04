# What is in this repository?
- We have a yolov5m trained on our self-collected Thai traffic sign dataset.
- We apply Conformal prediction on classification aspect of yolov5m.
- We apply Conformal risk control on object presenceness of yolov5m.
- These method should also work on other object detectors like RCNN, SSD, Detetron, etc. since conformal prediction does not make assumption about the underlying model. 

## Analyzing Confusion Matrix

<p align="left">
  <img src="https://github.com/PongsiriH/Traffic-Signs-detection-with-some-conformal-prediction/assets/127966686/534679d7-8347-4371-9660-d721c1e5c002" width="48%" alt="Confusion matrix with some mix-ups between speed limit signs and traffic signs vs. background" />
  <br>It turns out our model is a bit confused with speed limits and occasionally mistakes some background features for traffic signs. This mostly happens with traffic signs that were either missed or not included during our labeling process.
</p>

### Takeaways
**Speed Limit Signs**: It's clear we need to enrich our dataset with a wider variety of speed limit signs. Incorporating images captured from unique angles and under varied lighting conditions could significantly improve the model's ability to distinguish between them.
**Background Confusion**: A thorough review of our dataset is necessary to ensure no signs were overlooked or mislabeled. Additionally, integrating images that mimic potential traffic signs but are actually part of the background might enhance the model's discernment capabilities, reducing false positives.
**Model Tuning**: Experimenting with adjustments to the model's architecture or tuning its parameters could yield better results. Focusing on enhancements that improve the model's sensitivity to subtle differences among traffic signs may lead to more accurate predictions.

# Conformal prediction on classification aspect of yolov5m.
link to colab: https://colab.research.google.com/drive/1HSqgNwxLLsNKpAoIMJp1FfHbN-WIW-wU#scrollTo=0xUaMXqXPwlT

We present side-by-side comparisons to illustrate the effect of conformal prediction on the model's output.

**model is confidence and correct**
<p align="left">
  <img src="https://github.com/PongsiriH/Traffic-Signs-detection-with-some-conformal-prediction/assets/127966686/b7c29872-45cd-4d70-a616-ed60883406bc" width="48%" alt="Description 1" />
  <img src="https://github.com/PongsiriH/Traffic-Signs-detection-with-some-conformal-prediction/assets/127966686/59927797-246e-4def-9da1-267aeb702608" width="48%" alt="Description 2" />
  <br>Model confidently predicts the correct label in a single-class prediction set.
</p>

**model is confusing between speed limit signs**
<p align="left">
  <img src="https://github.com/PongsiriH/Traffic-Signs-detection-with-some-conformal-prediction/assets/127966686/c02ddb3a-a5cf-4d1f-9d02-7b09906a78ed" width="48%" alt="Description 3" />
  <img src="https://github.com/PongsiriH/Traffic-Signs-detection-with-some-conformal-prediction/assets/127966686/ea904ef4-56c1-4195-b2ad-86ef5c456af7" width="48%" alt="Description 4" />
  <br>Base prediction incorrectly predicts a "speed limit 50" sign as "speed limit 30", while conformal prediction corrects this with a set that includes "speed limit 50".
</p>

**model is underconfidence and prediction set is "wrong"**
<p align="left">
  <img src="https://github.com/PongsiriH/Traffic-Signs-detection-with-some-conformal-prediction/assets/127966686/d6080507-d10a-456a-b883-df0614611a6c" width="48%" alt="Description 7" />
  <img src="https://github.com/PongsiriH/Traffic-Signs-detection-with-some-conformal-prediction/assets/127966686/f2317c09-5ef3-497d-aae5-5db41ae65054" width="48%" alt="Description 8" />
  <br>Base model incorrectly predicts a "no parking" sign as "keep left," while the conformal prediction set reflects uncertainty with options for "keep left" and "keep right."
</p>

**nuance of model, dataset, etc.**
<p align="left">
  <img src="https://github.com/PongsiriH/Traffic-Signs-detection-with-some-conformal-prediction/assets/127966686/0456dfda-cf14-44db-a36e-0c64cd45923f" width="48%" alt="Description 5" />
  <img src="https://github.com/PongsiriH/Traffic-Signs-detection-with-some-conformal-prediction/assets/127966686/539d1c8c-0ed8-4332-84d3-e037a0687f33" width="48%" alt="Description 6" />
  <br>This image showcases 2 traffic signs from our dataset, both accurately identified by the model. Interestingly, the model generated 4 additional "background" predictions, identifying traffic signs we didn't include in our dataset. The prediction set for one of the recognized signs was unusually large, hinting at uncertainty, which is unexpected given the signs' apparent clarity. Among the predictions for out-of-domain signs, one displayed expected uncertainty with a prediction set of 2, while another was confidently but incorrectly identified with a prediction set of 1. This scenario illustrates that the effectiveness of conformal prediction is contingent on the base model's calibration.
</p>




# Conformal Risk Control: Controlling risk of missing pixels with respect to confidence thresold.
link to colab: https://colab.research.google.com/drive/19zJrENW3aFTL__NF5gGVxLH4SHoszHhU#scrollTo=YoK2xryOj1uY
