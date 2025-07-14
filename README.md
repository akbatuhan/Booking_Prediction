# Hotel Booking Cancellation Prediction

This project focuses on predicting hotel booking cancellations using a real-world dataset and applying both **classification** and **clustering** techniques in Python. The project includes thorough preprocessing, exploratory data analysis, model comparison, and customer segmentation.

---

## 📊 Project Objectives

- Predict whether a hotel booking will be **canceled** or **not** using classification models.
- Segment **non-canceling customers** into clusters and match them with canceling customers for customer behavior analysis and operational improvement.

---

## 🧠 Methods and Models

### 🔹 Classification Models Used:
- Logistic Regression
- Decision Tree
- Random Forest (🏆 Best accuracy: **89.14%**)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes

👉 Evaluation Metric: **Accuracy Score**  
👉 Best performing model: **Random Forest Classifier**

---

### 🔹 Clustering Models Used:
To group similar non-canceling customers:
- K-Means (🏆 Best clustering score)
- MiniBatch K-Means
- Agglomerative Clustering

👉 Evaluation Metrics:
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

Best clustering model based on all metrics: **K-Means**

---

## 🧹 Data Preprocessing

- Dataset: 36,258 records × 17 features  
- Features with no predictive value (e.g., `Booking_ID`, `Date_of_Reservation`) were removed.
- Highly imbalanced features were dropped (e.g., `Repeated`, `Car Parking Space`, etc.).
- Categorical variables were label encoded.
- Correlation matrix used for feature insights and selection.

---



## 📈 Results Summary

### 🔹 Classification:
- **Random Forest** achieved the highest accuracy.
- Imbalanced data was handled through preprocessing before training.

### 🔹 Clustering:
- **K-Means** provided the most distinct and valid clusters.
- Canceling customers were mapped to the most similar cluster of non-canceling ones for future marketing/policy actions.

---

## 💡 Example Use Case

> A customer predicted to **cancel** is matched to **Cluster 4**, which consists of customers who typically book earlier. Based on this insight, the hotel may send a reminder or promotion ~36 days before the stay to reduce the chance of cancellation.

---

## 📌 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 🔗 Dataset Source

- Kaggle Dataset: [Hotel Booking Cancellation Prediction – by Youssef Aboelwafa](https://www.kaggle.com/datasets/youssefaboelwafa/hotel-booking-cancellation-prediction)

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Silhouette Score Guide](https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c)
- [K-Means Clustering Tutorial](https://www.analyticsvidhya.com/blog/2021/07/an-introduction-to-logistic-regression/)
