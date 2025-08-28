# Machine Learning Loan Prediction System

This project is a machine learning-based loan prediction system that classifies whether a loan application should be approved or rejected based on applicant details.

## Project Structure

- `data/`: Contains the training and testing datasets.
- `model/`: Contains the saved machine learning model and label encoders.
- `templates/`: Contains the HTML templates for the Flask web application.
- `app.py`: The main Flask application file.
- `eda_and_preprocessing.ipynb`: Jupyter notebook for exploratory data analysis and preprocessing.
- `model_training.ipynb`: Jupyter notebook for training and evaluating multiple machine learning models.
- `hyperparameter_tuning.ipynb`: Jupyter notebook for tuning the best model's hyperparameters.
- `train_and_save_model.py`: Python script to train the final model and save it.
- `requirements.txt`: A list of the Python libraries required to run this project.
- `README.md`: This file.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- XGBoost
- Flask
- Jupyter Notebook

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\\Scripts\\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### 1. Exploratory Data Analysis and Model Training

You can explore the Jupyter notebooks to see the process of data analysis, preprocessing, model training, and evaluation:

- `eda_and_preprocessing.ipynb`
- `model_training.ipynb`
- `hyperparameter_tuning.ipynb`

To run the notebooks, you will need to have Jupyter Notebook installed (`pip install jupyter`).

### 2. Train and Save the Final Model

To train the final model and save it, run the following command:

```bash
python train_and_save_model.py
```

This will create the `loan_prediction_model.joblib` and `label_encoders.joblib` files in the `model/` directory.

### 3. Run the Web Application

To run the Flask web application, use the following command:

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`. You can open this URL in your web browser to use the loan prediction system.
