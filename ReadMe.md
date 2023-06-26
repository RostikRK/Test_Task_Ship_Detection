# Testing task documentation

In these project we have a few files:

- requirements.txt
- Data_Science_Task_Explanation.ipynb - I recommend to start from it as it has a whole process of coding in it with all the extension then I took a code from here and spread it by files
- EDA.ipynb - Explanatory data analysis
- utils.py - library with functions I used
- classification_train.py - file to train the classification model
- segmentation_train.py - file to train the segmentation model (the best one)
- inference.py - file to test how the segmentation model perform

Also, I provide you with the models I already trained you can find them in my google disk folder (https://drive.google.com/drive/folders/1J_SlnzurrwiB9O-6pRQ-kuU_BoH9mNq1?usp=sharing). There are:
- best_model_classification.h5 - my best classification model
- best_model_segment_resnet.h5 - the resnet model
- best_model_segment.h5 - my best efficient net model (spoiler: it gives 65% dice score)

Before running the code please run in CLI:

pip install -r requirements.txt

Also before running of any script be careful to specify the path to the file where you have stored it.
