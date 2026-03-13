# experience_prediction_svm
A machine learning project employing Support Vector Machines (SVM) for multiclass classification, designed to predict professional experience levels (Entry, Mid, Senior, Executive) using 2024 salary and demographic data.
# Steps to run the Project in Google Collab🚀
1. Download Dataset salary 2024.csv from repository.
2. Upload Dataset salary 2024.csv on Google Drive.
3. Start a New Notebook in Google Colab.
4. Mount Google Drive by running first code cell, then allow permission.
6. Click the Files icon and find the Dataset salary 2024.csv in the drive file.
7. Copy the path and paste it into read_csv() function.
8. Replace ... with your own path
   ```python df = pd.read_csv("/content/drive/MyDrive/...")```
9. Click on Runtime > Run all
10. Classification results will be available after the notebook loads the data and trains the SVM model.
# Model Summary
* Support Vector Machine (SVM) with hyperparameter tuning via GridSearchCV.
* Best Kernel: ```RBF``` (Non-linear boundary).
* Optimal Parameters: ```C=5, gamma='scale'```.
* Validation: ```3-fold Cross-Validation```.
* Optimization: Parallel processing (n_jobs=-1).
* The RBF kernel was selected due to the high-dimensional space created by ```One-Hot Encoding``` and the ```non-linear relationship``` between salary and seniority.
The RBF kernel was selected due to the high-dimensional space...

# Experimental Scenarios & Results

| Experimental Scenario | Accuracy | Key Observation |
| :--- | :---: | :--- |
| Complete removal of `job title` | 67% | Significant drop in predictive power; title is essential. |
| Preprocessing into `job group` | 69% | Improved classification for the dominant SE (Senior) class. |
| Retaining original `job title` | **71%** | **Best performance**; better capture of EN and EX classes. |
# Best Performance Confusion Matrix
<img width="448" height="470" alt="image" src="https://github.com/user-attachments/assets/4cafa022-c4ef-454f-a5b2-ccacd4d0cbe5" />
