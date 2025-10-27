# 🎓 Student Performance Prediction System

A comprehensive Streamlit web application that predicts student academic performance using machine learning models. The system provides both individual and batch predictions for student results and marks.

## 🌟 Features

### 🤖 Machine Learning Models
- **Result Prediction Model**: Predicts whether a student will Pass (1) or Fail (0)
- **Marks Prediction Model**: Predicts numerical marks for students

### 📊 Prediction Types
1. **Individual Result Prediction** (Pass/Fail)
2. **Individual Marks Prediction** (Numerical marks)
3. **Batch Analysis** (10 students with historical data and future predictions)

### 🎯 Key Functionalities

#### Individual Predictions
- Real-time predictions for single students
- Interactive input forms with sliders and select boxes
- Confidence scores and performance categories
- Visual progress bars and performance indicators

#### Batch Analysis
- **10 Sample Students** with 3 years of historical data
- **One-click predictions** for next academic year
- **Dual predictions**: Both results AND marks
- **Comprehensive analytics** and performance distribution
- **Multiple view modes**: Summary, Detailed Marks, Full Details

### 📈 Analytics & Insights
- Pass/Fail statistics and probabilities
- Performance categorization (Excellent, Good, Average, Needs Improvement)
- Subject-wise marks analysis
- Historical trend visualization
- Risk assessment for at-risk students

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Step 1: Clone/Download the Project
```bash
git clone <repository-url>
cd student-performance-predictor
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Model Files
Ensure you have both trained model files in the project directory:
- `student_result_model.pkl` - Result prediction model
- `student_marks_model.pkl` - Marks prediction model

### Step 4: Run the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
student-performance-predictor/
│
├── app.py                 # Main Streamlit application
├── student_result_model.pkl    # Result prediction model
├── student_marks_model.pkl     # Marks prediction model
├── README.md              # This documentation
└── requirements.txt       # Python dependencies
```

## 🎮 How to Use

### 1. Individual Result Prediction
- Navigate to the "Result Prediction" tab
- Fill in student details:
  - Year, Gender, Attendance percentage
  - Study hours per day
  - Extra activity level
  - Subject marks (Math, Science, English, Social Studies)
- Click "Predict Result" to get Pass/Fail prediction

### 2. Individual Marks Prediction
- Go to the "Marks Prediction" tab
- Provide student information:
  - Year, Gender, Attendance
  - Study hours, Previous year marks
  - Parent education level
- Click "Predict Marks" for numerical marks prediction

### 3. Batch Analysis
- Select the "Batch Analysis" tab
- View 10 sample students with 3 years of historical data
- Select individual students to see detailed history
- Click "Predict Next Year Results & Marks" for batch predictions
- Explore different views: Summary, Detailed Marks, Student Details
- Download complete predictions as CSV

## 🔧 Model Input Formats

### Result Prediction Model Expects:
```python
{
    'Year': int,
    'Gender': ['Male', 'Female'],
    'Attendance': int (0-100),
    'Study_Hours': float,
    'Extra_Activity': ['Low', 'Medium', 'High'],
    'Math': int (0-100),
    'Science': int (0-100),
    'English': int (0-100),
    'Social': int (0-100)
}
```

### Marks Prediction Model Expects:
```python
{
    'year': str,
    'gender': ['Male', 'Female'],
    'attendance': int (0-100),
    'study_hours': float,
    'previous_year_marks': int (0-100),
    'parent_education': ['High School', 'Graduate', 'Post Graduate', 'Doctorate']
}
```

## 📊 Output Interpretation

### Result Predictions
- **PASS ✅**: Student is predicted to pass
- **FAIL ❌**: Student is predicted to fail
- **Pass Probability**: Confidence percentage (0-100%)

### Marks Predictions
- **Excellent 🎉**: 80+ marks
- **Good 📚**: 60-79 marks
- **Average 💪**: 40-59 marks
- **Needs Improvement 📝**: Below 40 marks

### Performance Categories
- **Excellent** (80-100 marks): Outstanding performance
- **Good** (60-79 marks): Solid performance
- **Average** (40-59 marks): Room for improvement
- **Needs Improvement** (0-39 marks): Requires attention

## 🛠️ Technical Details

### Built With
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Joblib**: Model serialization and loading
- **Scikit-learn**: Machine learning models (assumed)

### Features
- **Responsive Design**: Works on desktop and mobile
- **Real-time Predictions**: Instant results
- **Data Caching**: Optimized performance
- **Error Handling**: Graceful error management
- **Export Functionality**: Download predictions as CSV

## 📈 Use Cases

### For Educators
- Identify at-risk students early
- Plan targeted interventions
- Monitor class performance trends
- Generate performance reports

### For Students
- Self-assessment and goal setting
- Understand performance factors
- Identify areas for improvement

### For Administrators
- Institutional performance analysis
- Resource allocation planning
- Curriculum improvement insights

**Note**: This application requires pre-trained machine learning models (`student_result_model.pkl` and `student_marks_model.pkl`) to function properly. Ensure these files are in the same directory as the application.