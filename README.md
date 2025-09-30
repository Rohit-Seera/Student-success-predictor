# Student Success Predictor 🎓

A machine learning project that predicts student academic success using various academic and personal factors. The model analyzes multiple student attributes including grades, attendance, participation, and socio-economic factors to predict whether a student will pass or fail.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

This project aims to help educational institutions identify at-risk students early in their academic journey. By leveraging machine learning algorithms, specifically Logistic Regression, the model analyzes various student characteristics and academic performance indicators to predict academic success.

### Problem Statement
Educational institutions face challenges in identifying students who might struggle academically. Early identification of at-risk students allows for timely interventions, support programs, and resources allocation to improve student outcomes.

### Solution
Our Student Success Predictor provides:
- **Power Mode**: Comprehensive prediction using all available features
- **Quick Mode**: Fast prediction using key indicators
- Interactive interface with real-time predictions
- Visual performance analysis through confusion matrix

## Features 

### Prediction Modes
- **🚀 Power Mode**: Uses all available student features for comprehensive analysis
- **💻 Quick Mode**: Focuses on key features for quick assessments
  - Previous Grades
  - Attendance
  - Class Participation
  - Motivation
  - Parental Involvement
  - Stress Levels
  - Professor Quality

### Key Capabilities
- Binary classification (Pass/Fail prediction)
- Interactive command-line interface
- Data preprocessing and feature engineering
- Model performance visualization
- Real-time prediction capabilities

## Dataset

The model uses the `University_Student_Dataset_Final_with_Pass_Fail.csv` dataset containing student information across multiple categories:

### Feature Categories

#### Ordinal Features
- Previous_Grades, Parental_Education, Attendance
- Class_Participation, Financial_Status, Parental_Involvement
- Motivation, Self_Esteem, Stress_Levels
- School_Environment, Professor_Quality, Sleep_Patterns
- Nutrition, Physical_Activity, Lack_of_Interest

#### Nominal Features
- Gender, Major, School_Type, Educational_Resources
- Extracurricular_Activities, Screen_Time, Educational_Tech_Use
- Peer_Group, Bullying, Study_Space
- Learning_Style, Tutoring, Mentoring, Sports_Participation

## Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/student-success-predictor.git
   cd student-success-predictor
   ```

2. **Install required packages**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

3. **Download the dataset**
   - Ensure you have the `University_Student_Dataset_Final_with_Pass_Fail.csv` file
   - Place it in the same directory as the Python script

## Usage

### Running the Application

```bash
python studentsuccess_predictor.py
```

### Interactive Menu Options

1. **Power Mode 🚀**: Enter values for all features
2. **Quick Mode ⚡**: Enter values for key features only
3. **Exit**: Terminate the program

### Example Usage

```
------------------------------
Welcome to student success predictor
------------------------------
Select the option 1. power mode 🚀 2.quick mode 💻 , 3.exit
Enter your choice: (1/2/3): 2

You are in Quick Mode ⚡
Please enter the following Information:
Enter value for Previous_Grades: 85
Enter value for Attendance: 90
Enter value for Class_Participation: 80
...

🎓 Prediction Result:
✅ You are likely to Pass! 🎉
```

## Model Performance

The model uses **Logistic Regression** with the following configuration:
- **Algorithm**: Logistic Regression
- **Class Weight**: Balanced (to handle class imbalance)
- **Max Iterations**: 1000
- **Random State**: 42 (for reproducibility)

### Performance Metrics
- Classification Report with precision, recall, and F1-score
- Confusion Matrix visualization
- Model accuracy assessment

### Data Preprocessing
- **Label Encoding** for ordinal features
- **One-Hot Encoding** for nominal features
- **Standard Scaling** for feature normalization
- **Train-Test Split**: 80-20 ratio

## Project Structure

```
student-success-predictor/
│
├── studentsucces_predictor.py    # Main application file
├── University_Student_Dataset_Final_with_Pass_Fail.csv  # Dataset
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── images/                       # Screenshots and visualizations
    └── confusion_matrix.png      # Model performance visualization
```

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the Project**
2. **Create your Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your Changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Areas for Improvement
- Add more machine learning algorithms (Random Forest, SVM, Neural Networks)
- Implement cross-validation
- Create a web-based interface
- Add feature importance analysis
- Include more comprehensive error handling

## Contact

Rohit Meena - rohitseera777.gmail@example.com

Project Link: [https://github.com/Rohit-Seera/Student-success-predictor](https://github.com/Rohit-Seera/Student-success-predictor)

## Acknowledgments

- Thanks to the creators of the University Student Dataset
- Scikit-learn community for excellent documentation
- Educational institutions working on student success initiatives

---

**⭐ If you found this project helpful, please consider giving it a star!**
