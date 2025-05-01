# Sustainable Material Selection Using Machine Learning

## Project Overview
This project aims to develop a machine learning model to predict the sustainability rating of construction materials based on their characteristics such as cost, strength, durability, environmental impact, and recyclability. The goal is to assist architects, engineers, and construction managers in selecting eco-friendly materials that promote sustainable building practices and reduce the environmental footprint of construction projects.

## Features
- Predict sustainability rating (High, Medium, Low) of construction materials based on input features.
- Interactive Streamlit web app for user input and real-time prediction.
- Data visualization including bar charts, scatter plots, box plots, confusion matrix, feature importance, and ROC curve.
- Uses a Random Forest classification model with preprocessing pipeline for accurate predictions.

## Dataset
The dataset consists of synthetic data describing various construction materials with the following fields:
- Material Type (Concrete, Steel, Wood)
- Cost (numeric)
- Strength (MPa)
- Durability (years)
- Environmental Impact (Low, Medium, High)
- Recyclability (Yes, No)
- Sustainability Rating (High, Medium, Low) - Target variable

## Installation and Setup

### Prerequisites
- Python 3.7 or above
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Running the App
1. Clone the repository:
```bash
git clone https://github.com/JAYASIMMA/Machine-Learning-Models-for-Sustainable-Material-Selection-in-Construction.git
cd Machine-Learning-Models-for-Sustainable-Material-Selection-in-Construction
cd project
```
2. Run the Streamlit application:
```bash
streamlit run app.py
```
3. The app will launch in your default browser.

## Usage
- Input material properties into the Streamlit application.
- Click the "Predict Sustainability Rating" button to see the model's prediction.
- Explore various data visualizations and performance metrics via the interface.

## File Structure
- `app.py` - Streamlit app for user interaction and visualization.
- `model_training.py` - Script for training the machine learning model and saving it.
- `sustainable_materials_dataset.csv` - Synthetic dataset used for training and evaluation.
- `sustainable_material_model.pkl` - Trained model saved for deployment.

## Metrics Evaluated
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve

##Screenshots
![Screenshot 2025-05-01 055329](https://github.com/user-attachments/assets/b2be1814-83bb-4c9d-96a5-764f0238fe2a)
![Screenshot 2025-05-01 063509](https://github.com/user-attachments/assets/987262af-f744-4357-a752-f2a814eb625b)
![Screenshot 2025-05-01 063829](https://github.com/user-attachments/assets/140abf38-2e6b-4799-9237-90c706d4b890)
![Screenshot 2025-05-01 063725](https://github.com/user-attachments/assets/a22a4ca4-e055-45b0-ba48-d16ba202d0a6)
![Screenshot 2025-05-01 063657](https://github.com/user-attachments/assets/07049b41-db3a-4b1f-9dc6-73dc2b45118b)
![Screenshot 2025-05-01 063639](https://github.com/user-attachments/assets/2a7c00bc-68c6-45a6-bfad-cc72803dfa9d)
![Screenshot 2025-05-01 063607](https://github.com/user-attachments/assets/7e58cdfb-b975-47c4-9ecc-1535f607a288)


## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your improvements.

## License
This project is licensed under the MIT License.

## Contact
For any inquiries or support, please contact Jayasimma D at jayasimmamomdad@gmail.com.

---
*This project demonstrates a practical application of AI and machine learning to enable sustainable decision-making in construction.*

```

