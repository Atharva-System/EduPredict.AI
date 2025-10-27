import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime
import random

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Title and description
st.title("🎓 Student Performance Prediction System")
st.markdown("""
This application predicts both student exam results (Pass/Fail) and expected marks using machine learning models.
""")

# Check if model files exist
if not os.path.exists("student_result_model.pkl") or not os.path.exists("student_marks_model.pkl"):
    st.error("⚠️ Model files not found! Please ensure both 'student_result_model.pkl' and 'student_marks_model.pkl' are in the same directory.")
    st.stop()

# Load models (with caching to avoid reloading on every interaction)
@st.cache_resource
def load_models():
    try:
        result_model = joblib.load("student_result_model.pkl")
        marks_model = joblib.load("student_marks_model.pkl")
        return result_model, marks_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

result_model, marks_model = load_models()

if result_model is None or marks_model is None:
    st.stop()

# Generate sample student data for last 3 years
def generate_sample_students():
    students = []
    for i in range(1, 11):
        base_attendance = random.randint(75, 95)
        base_study_hours = round(random.uniform(2.0, 5.0), 1)
        base_math = random.randint(65, 90)
        base_science = random.randint(60, 88)
        base_english = random.randint(70, 92)
        base_social = random.randint(65, 85)
        
        student = {
            'student_id': f'STU{1000 + i}',
            'name': f'Student {i}',
            'gender': random.choice(['Male', 'Female']),
            'base_attendance': base_attendance,
            'base_study_hours': base_study_hours,
            'base_math': base_math,
            'base_science': base_science,
            'base_english': base_english,
            'base_social': base_social,
            'extra_activity': random.choice(['Low', 'Medium', 'High']),
            'parent_education': random.choice(['High School', 'Graduate', 'Post Graduate', 'Doctorate'])
        }
        students.append(student)
    return students

def generate_historical_performance(students):
    historical_data = []
    current_year = datetime.now().year
    
    for student in students:
        for year_offset in range(3, 0, -1):  # Years 3, 2, 1 ago
            year = current_year - year_offset
            
            # Add some variation to simulate performance changes
            attendance_variation = random.randint(-5, 5)
            study_hours_variation = random.uniform(-0.5, 0.5)
            marks_variation = random.randint(-8, 8)
            
            # Calculate total marks for historical data
            math_marks = max(50, min(100, student['base_math'] + marks_variation))
            science_marks = max(50, min(100, student['base_science'] + marks_variation))
            english_marks = max(50, min(100, student['base_english'] + marks_variation))
            social_marks = max(50, min(100, student['base_social'] + marks_variation))
            total_marks = (math_marks + science_marks + english_marks + social_marks) / 4
            
            historical_data.append({
                'student_id': student['student_id'],
                'name': student['name'],
                'year': year,
                'gender': student['gender'],
                'attendance': max(60, min(100, student['base_attendance'] + attendance_variation)),
                'study_hours': round(max(1.0, min(8.0, student['base_study_hours'] + study_hours_variation)), 1),
                'math': math_marks,
                'science': science_marks,
                'english': english_marks,
                'social': social_marks,
                'total_marks': round(total_marks, 2),
                'extra_activity': student['extra_activity'],
                'parent_education': student['parent_education'],
                'actual_result': 1 if total_marks >= 40 else 0  # Simulate historical results
            })
    
    return pd.DataFrame(historical_data)

def predict_future_performance(historical_df, result_model, marks_model):
    predictions = []
    current_year = datetime.now().year
    next_year = current_year + 1
    
    # Get unique students
    unique_students = historical_df['student_id'].unique()
    
    for student_id in unique_students:
        student_data = historical_df[historical_df['student_id'] == student_id]
        latest_year = student_data['year'].max()
        latest_data = student_data[student_data['year'] == latest_year].iloc[0]
        
        # Calculate trends (simple average of last 3 years)
        avg_attendance = student_data['attendance'].mean()
        avg_study_hours = student_data['study_hours'].mean()
        avg_math = student_data['math'].mean()
        avg_science = student_data['science'].mean()
        avg_english = student_data['english'].mean()
        avg_social = student_data['social'].mean()
        avg_total_marks = student_data['total_marks'].mean()
        
        # Prepare prediction data for next year with slight improvement trend
        pred_attendance = min(100, avg_attendance + random.randint(0, 3))
        pred_study_hours = round(min(8.0, avg_study_hours + random.uniform(0, 0.3)), 1)
        pred_math = min(100, avg_math + random.randint(0, 5))
        pred_science = min(100, avg_science + random.randint(0, 5))
        pred_english = min(100, avg_english + random.randint(0, 5))
        pred_social = min(100, avg_social + random.randint(0, 5))
        
        # PREDICT RESULT (Pass/Fail)
        result_prediction_data = pd.DataFrame({
            'Year': [next_year],
            'Gender': [latest_data['gender']],
            'Attendance': [pred_attendance],
            'Study_Hours': [pred_study_hours],
            'Extra_Activity': [latest_data['extra_activity']],
            'Math': [pred_math],
            'Science': [pred_science],
            'English': [pred_english],
            'Social': [pred_social]
        })
        
        try:
            result_prediction = result_model.predict(result_prediction_data)[0]
            result_prob = None
            
            if hasattr(result_model, 'predict_proba'):
                probabilities = result_model.predict_proba(result_prediction_data)
                result_prob = probabilities[0][1]  # Probability of passing
        except Exception as e:
            st.error(f"Error in result prediction for {student_id}: {str(e)}")
            result_prediction = 1 if avg_total_marks >= 40 else 0
            result_prob = 0.8 if result_prediction == 1 else 0.2
        
        # PREDICT MARKS using marks model
        try:
            marks_prediction_data = pd.DataFrame([{
                'year': str(next_year),
                'gender': latest_data['gender'],
                'attendance': pred_attendance,
                'study_hours': pred_study_hours,
                'previous_year_marks': avg_total_marks,
                'parent_education': latest_data['parent_education']
            }])
            marks_prediction = marks_model.predict(marks_prediction_data)[0]
            marks_prediction = max(0, min(100, marks_prediction))  # Ensure marks are within bounds
        except Exception as e:
            st.error(f"Error in marks prediction for {student_id}: {str(e)}")
            # Fallback calculation if marks model fails
            marks_prediction = (pred_math + pred_science + pred_english + pred_social) / 4
        
        # Calculate subject-wise marks predictions
        subject_marks_prediction = {
            'math': pred_math,
            'science': pred_science,
            'english': pred_english,
            'social': pred_social
        }
        
        # Performance category
        if marks_prediction >= 80:
            performance_category = "Excellent 🎉"
        elif marks_prediction >= 60:
            performance_category = "Good 📚"
        elif marks_prediction >= 40:
            performance_category = "Average 💪"
        else:
            performance_category = "Needs Improvement 📝"
        
        predictions.append({
            'student_id': student_id,
            'name': latest_data['name'],
            'predicted_year': next_year,
            'predicted_result': 'PASS' if result_prediction == 1 else 'FAIL',
            'pass_probability': round(result_prob * 100, 2) if result_prob else 'N/A',
            'predicted_total_marks': round(marks_prediction, 2),
            'performance_category': performance_category,
            'predicted_math': pred_math,
            'predicted_science': pred_science,
            'predicted_english': pred_english,
            'predicted_social': pred_social,
            'attendance': pred_attendance,
            'study_hours': pred_study_hours,
            'parent_education': latest_data['parent_education'],
            'extra_activity': latest_data['extra_activity']
        })
    
    return pd.DataFrame(predictions)

# Create tabs for different prediction types
tab1, tab2, tab3 = st.tabs(["📊 Result Prediction (Pass/Fail)", "📈 Marks Prediction", "👥 Batch Analysis - 10 Students"])

with tab1:
    st.header("Student Result Prediction")
    st.markdown("Predict whether a student will **Pass (1)** or **Fail (0)**")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.number_input("Year", min_value=2000, max_value=2030, value=2024, key="result_year")
        gender = st.selectbox("Gender", ["Male", "Female"], key="result_gender")
        attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=85, key="result_attendance")
        study_hours = st.slider("Study Hours (per day)", min_value=0.0, max_value=10.0, value=3.5, step=0.5, key="result_study_hours")
    
    with col2:
        extra_activity = st.selectbox("Extra Activity Level", ["Low", "Medium", "High"], key="result_activity")
        math_marks = st.slider("Math Marks", min_value=0, max_value=100, value=78, key="result_math")
        science_marks = st.slider("Science Marks", min_value=0, max_value=100, value=72, key="result_science")
        english_marks = st.slider("English Marks", min_value=0, max_value=100, value=80, key="result_english")
        social_marks = st.slider("Social Studies Marks", min_value=0, max_value=100, value=76, key="result_social")
    
    # Prediction button for result model
    if st.button("Predict Result", key="predict_result"):
        try:
            # Prepare data for result prediction
            new_data = pd.DataFrame({
                'Year': [year],
                'Gender': [gender],
                'Attendance': [attendance],
                'Study_Hours': [study_hours],
                'Extra_Activity': [extra_activity],
                'Math': [math_marks],
                'Science': [science_marks],
                'English': [english_marks],
                'Social': [social_marks]
            })
            
            # Make prediction
            prediction = result_model.predict(new_data)
            result = "PASS ✅" if prediction[0] == 1 else "FAIL ❌"
            
            # Display result
            st.success(f"**Prediction Result: {result}**")
            
            # Show confidence or additional info if available
            if hasattr(result_model, 'predict_proba'):
                probabilities = result_model.predict_proba(new_data)
                st.info(f"Confidence: Pass {probabilities[0][1]:.2%}, Fail {probabilities[0][0]:.2%}")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

with tab2:
    st.header("Student Marks Prediction")
    st.markdown("Predict the expected marks for a student")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        year_marks = st.number_input("Year", min_value=2000, max_value=2030, value=2023, key="marks_year")
        gender_marks = st.selectbox("Gender", ["Male", "Female"], key="marks_gender")
        attendance_marks = st.slider("Attendance (%)", min_value=0, max_value=100, value=85, key="marks_attendance")
    
    with col2:
        study_hours_marks = st.slider("Study Hours (per day)", min_value=0.0, max_value=10.0, value=3.5, step=0.5, key="marks_study_hours")
        previous_marks = st.slider("Previous Year Marks", min_value=0, max_value=100, value=70, key="marks_previous")
        parent_education = st.selectbox("Parent Education", 
                                       ["High School", "Graduate", "Post Graduate", "Doctorate"], 
                                       key="marks_parent_edu")
    
    # Prediction button for marks model
    if st.button("Predict Marks", key="predict_marks"):
        try:
            # Prepare data for marks prediction
            sample = pd.DataFrame([{
                'year': str(year_marks),
                'gender': gender_marks,
                'attendance': attendance_marks,
                'study_hours': study_hours_marks,
                'previous_year_marks': previous_marks,
                'parent_education': parent_education
            }])
            
            # Make prediction
            predicted_marks = marks_model.predict(sample)
            rounded_marks = round(predicted_marks[0], 2)
            
            # Display result with visual indicator
            st.success(f"**Predicted Marks: {rounded_marks}**")
            
            # Add a visual progress bar
            st.progress(min(rounded_marks / 100, 1.0))
            st.caption(f"{rounded_marks}/100")
            
            # Performance interpretation
            if rounded_marks >= 80:
                st.balloons()
                st.success("🎉 Excellent performance predicted!")
            elif rounded_marks >= 60:
                st.info("📚 Good performance predicted!")
            elif rounded_marks >= 40:
                st.warning("💪 Average performance - room for improvement!")
            else:
                st.error("📝 Needs attention and improvement!")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

with tab3:
    st.header("📊 Batch Analysis - 10 Students")
    st.markdown("""
    **View historical performance of 10 students over the last 3 years and predict their results AND marks for the next year.**
    """)
    
    # Generate sample data
    if 'historical_data' not in st.session_state:
        students = generate_sample_students()
        st.session_state.historical_data = generate_historical_performance(students)
    
    # Display historical data
    st.subheader("📋 Historical Performance (Last 3 Years)")
    
    # Let user select a student to view detailed history
    student_ids = st.session_state.historical_data['student_id'].unique()
    selected_student = st.selectbox("Select student to view detailed history:", student_ids)
    
    # Show detailed history for selected student
    student_history = st.session_state.historical_data[st.session_state.historical_data['student_id'] == selected_student]
    
    # Display historical performance with metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Marks", f"{student_history['total_marks'].mean():.1f}")
    with col2:
        st.metric("Average Attendance", f"{student_history['attendance'].mean():.1f}%")
    with col3:
        st.metric("Pass Rate", f"{(student_history['actual_result'].sum() / len(student_history)) * 100:.1f}%")
    with col4:
        st.metric("Study Hours Avg", f"{student_history['study_hours'].mean():.1f}")
    
    st.dataframe(student_history[['year', 'attendance', 'study_hours', 'math', 'science', 'english', 'social', 'total_marks', 'actual_result']], 
                use_container_width=True)
    
    # Predict future performance button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🎯 Predict Next Year Results & Marks", type="primary", use_container_width=True):
            with st.spinner("Generating predictions for all students..."):
                predictions_df = predict_future_performance(st.session_state.historical_data, result_model, marks_model)
                st.session_state.predictions = predictions_df
        
    # Display predictions if available
    if 'predictions' in st.session_state:
        st.subheader("🎓 Next Year Predictions - Results & Marks")
        
        # Create tabs for different views of predictions
        pred_tab1, pred_tab2, pred_tab3 = st.tabs(["📊 Summary View", "📈 Detailed Marks", "📋 Student Details"])
        
        with pred_tab1:
            # Summary view with key metrics
            summary_df = st.session_state.predictions[[
                'student_id', 'name', 'predicted_result', 'pass_probability', 
                'predicted_total_marks', 'performance_category', 'attendance', 'study_hours'
            ]]
            
            st.dataframe(
                summary_df,
                use_container_width=True,
                column_config={
                    'student_id': 'Student ID',
                    'name': 'Name',
                    'predicted_result': st.column_config.TextColumn(
                        'Predicted Result',
                        help="Predicted result for next year"
                    ),
                    'pass_probability': st.column_config.ProgressColumn(
                        'Pass Probability %',
                        help="Probability of passing",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    'predicted_total_marks': st.column_config.ProgressColumn(
                        'Predicted Marks',
                        help="Predicted total marks",
                        format="%.1f",
                        min_value=0,
                        max_value=100,
                    ),
                    'performance_category': 'Performance',
                    'attendance': 'Attendance %',
                    'study_hours': 'Study Hours'
                }
            )
        
        with pred_tab2:
            # Detailed marks view
            marks_df = st.session_state.predictions[[
                'student_id', 'name', 'predicted_total_marks', 'performance_category',
                'predicted_math', 'predicted_science', 'predicted_english', 'predicted_social'
            ]]
            
            # Add marks progress bars
            st.dataframe(
                marks_df,
                use_container_width=True,
                column_config={
                    'student_id': 'Student ID',
                    'name': 'Name',
                    'predicted_total_marks': st.column_config.ProgressColumn(
                        'Total Marks',
                        format="%.1f",
                        min_value=0,
                        max_value=100,
                    ),
                    'performance_category': 'Performance',
                    'predicted_math': st.column_config.ProgressColumn(
                        'Math',
                        format="%.1f",
                        min_value=0,
                        max_value=100,
                    ),
                    'predicted_science': st.column_config.ProgressColumn(
                        'Science',
                        format="%.1f",
                        min_value=0,
                        max_value=100,
                    ),
                    'predicted_english': st.column_config.ProgressColumn(
                        'English',
                        format="%.1f",
                        min_value=0,
                        max_value=100,
                    ),
                    'predicted_social': st.column_config.ProgressColumn(
                        'Social',
                        format="%.1f",
                        min_value=0,
                        max_value=100,
                    )
                }
            )
        
        with pred_tab3:
            # Full details view
            st.dataframe(
                st.session_state.predictions,
                use_container_width=True
            )
        
        # Add comprehensive statistics
        st.subheader("📈 Batch Performance Analytics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        predictions_df = st.session_state.predictions
        pass_count = len(predictions_df[predictions_df['predicted_result'] == 'PASS'])
        fail_count = len(predictions_df[predictions_df['predicted_result'] == 'FAIL'])
        avg_marks = predictions_df['predicted_total_marks'].mean()
        avg_attendance = predictions_df['attendance'].mean()
        avg_study_hours = predictions_df['study_hours'].mean()
        
        with col1:
            st.metric("🎯 Passing Students", f"{pass_count}/10", f"{pass_count * 10}%")
        with col2:
            st.metric("⚠️ At-Risk Students", f"{fail_count}/10", f"{fail_count * 10}%")
        with col3:
            st.metric("📊 Average Marks", f"{avg_marks:.1f}")
        with col4:
            st.metric("📅 Average Attendance", f"{avg_attendance:.1f}%")
        with col5:
            st.metric("⏰ Study Hours", f"{avg_study_hours:.1f}")
        
        # Performance distribution
        st.subheader("📊 Performance Distribution")
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        excellent_count = len(predictions_df[predictions_df['performance_category'].str.contains('Excellent')])
        good_count = len(predictions_df[predictions_df['performance_category'].str.contains('Good')])
        average_count = len(predictions_df[predictions_df['performance_category'].str.contains('Average')])
        needs_improvement_count = len(predictions_df[predictions_df['performance_category'].str.contains('Needs Improvement')])
        
        with perf_col1:
            st.metric("Excellent 🎉", excellent_count)
        with perf_col2:
            st.metric("Good 📚", good_count)
        with perf_col3:
            st.metric("Average 💪", average_count)
        with perf_col4:
            st.metric("Needs Help 📝", needs_improvement_count)
        
        # Download button for predictions
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Predictions as CSV",
            data=csv,
            file_name=f"student_complete_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Sidebar with additional information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Models Available:**
    - 🎯 **Result Prediction**: Predicts Pass (1) or Fail (0)
    - 📊 **Marks Prediction**: Predicts numerical marks
    - 👥 **Batch Analysis**: Predicts results AND marks for multiple students
    
    **How to use:**
    1. Select the appropriate tab
    2. Fill in the student details
    3. Click the prediction button
    4. View the results and insights
    """)
    
    st.header("📊 Model Information")
    st.metric("Result Model", "Loaded ✅")
    st.metric("Marks Model", "Loaded ✅")
    
    # Display sample input formats
    with st.expander("📋 Expected Input Formats"):
        st.markdown("""
        **Result Model expects:**
        - Year, Gender, Attendance, Study_Hours
        - Extra_Activity, Math, Science, English, Social
        
        **Marks Model expects:**
        - year, gender, attendance, study_hours
        - previous_year_marks, parent_education
        """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Student Performance Prediction System")