# Student Habits Performance Pipeline Documentation

## Pipeline Overview
This pipeline preprocesses the Student Habits Performance dataset to prepare it for binary classification modeling. It handles categorical encoding, outlier detection and treatment, feature scaling, and missing value imputation to predict student academic performance (high vs. low performance based on exam scores).

## Target Variable Creation
- **Binary Target**: `high_performance` created from `exam_score` using median threshold
- **Rationale**: Converts continuous exam scores into a balanced binary classification problem

## Step-by-Step Design Choices

### 1. Gender Mapping (map_gender)
**Transformer**: `CustomMappingTransformer('gender', {'Male': 0, 'Female': 1, 'Other': 2})`
- **Design Choice**: Categorical encoding with three categories
- **Rationale**: Preserves all gender categories while converting to numerical format for ML algorithms

### 2. Part-Time Job Mapping (map_part_time_job)
**Transformer**: `CustomMappingTransformer('part_time_job', {'No': 0, 'Yes': 1})`
- **Design Choice**: Binary encoding with employment status
- **Rationale**: Simple binary feature that may correlate with study time and academic performance

### 3. Extracurricular Participation Mapping (map_extracurricular)
**Transformer**: `CustomMappingTransformer('extracurricular_participation', {'No': 0, 'Yes': 1})`
- **Design Choice**: Binary encoding for activity participation
- **Rationale**: Captures student engagement outside academics, which may impact performance

### 4. Diet Quality Mapping (map_diet_quality)
**Transformer**: `CustomMappingTransformer('diet_quality', {'Poor': 0, 'Fair': 1, 'Good': 2})`
- **Design Choice**: Ordinal encoding preserving quality hierarchy
- **Rationale**: Diet quality has natural ordering that may correlate with cognitive performance and study habits

### 5. Internet Quality Mapping (map_internet_quality)
**Transformer**: `CustomMappingTransformer('internet_quality', {'Poor': 0, 'Average': 1, 'Good': 2})`
- **Design Choice**: Ordinal encoding for connectivity quality
- **Rationale**: Internet quality affects access to educational resources and online learning effectiveness

### 6. Parental Education Mapping (map_parental_education)
**Transformer**: `CustomMappingTransformer('parental_education_level', {'None': 0, 'High School': 1, 'Bachelor': 2, 'Master': 3})`
- **Design Choice**: Ordinal encoding following education hierarchy
- **Rationale**: Parental education strongly correlates with student academic support and expectations

### 7. Age Outlier Treatment (tukey_age)
**Transformer**: `CustomTukeyTransformer(target_column='age', fence='outer')`
- **Design Choice**: Tukey method with outer fence for extreme outliers
- **Rationale**: Age should be relatively consistent for students; outer fence preserves most legitimate variation

### 8. Study Hours Outlier Treatment (tukey_study_hours)
**Transformer**: `CustomTukeyTransformer(target_column='study_hours_per_day', fence='outer')`
- **Design Choice**: Outer fence for study time outliers
- **Rationale**: Some students may legitimately study many hours; outer fence removes only extreme values (>24 hours/day)

### 9. Social Media Hours Outlier Treatment (tukey_social_media)
**Transformer**: `CustomTukeyTransformer(target_column='social_media_hours', fence='outer')`
- **Design Choice**: Outer fence for social media usage
- **Rationale**: High variation expected in social media usage; removes only impossible values

### 10. Netflix Hours Outlier Treatment (tukey_netflix)
**Transformer**: `CustomTukeyTransformer(target_column='netflix_hours', fence='outer')`
- **Design Choice**: Outer fence for entertainment consumption
- **Rationale**: Streaming habits vary widely; outer fence removes only extreme outliers

### 11. Attendance Percentage Outlier Treatment (tukey_attendance)
**Transformer**: `CustomTukeyTransformer(target_column='attendance_percentage', fence='outer')`
- **Design Choice**: Outer fence for attendance rates
- **Rationale**: Attendance should be between 0-100%; removes data entry errors

### 12. Sleep Hours Outlier Treatment (tukey_sleep)
**Transformer**: `CustomTukeyTransformer(target_column='sleep_hours', fence='outer')`
- **Design Choice**: Outer fence for sleep duration
- **Rationale**: Sleep has biological constraints; removes physiologically impossible values

### 13-18. Feature Scaling (scale_*)
**Transformers**: `CustomRobustTransformer` applied to all continuous features
- **Design Choice**: Robust scaling for all numerical features
- **Rationale**: 
 - Robust to remaining outliers after Tukey treatment
 - Uses median and IQR instead of mean and standard deviation
 - Prevents features with larger scales from dominating distance-based algorithms

### 19. Column Removal (drop_columns)
**Transformer**: `CustomDropColumnsTransformer(['exam_score', 'student_id'], action='drop')`
- **Design Choice**: Remove identifier and original target
- **Rationale**: 
 - `student_id` is just an identifier with no predictive value
 - `exam_score` is the source of our binary target, keeping it would cause data leakage

### 20. Missing Value Imputation (impute)
**Transformer**: `CustomKNNTransformer(n_neighbors=5)`
- **Design Choice**: KNN imputation with 5 neighbors
- **Rationale**:
 - Uses relationships between student characteristics to estimate missing values
 - k=5 balances between local patterns and overfitting
 - More sophisticated than mean/median imputation for complex relationships

## Pipeline Execution Order Rationale

1. **Categorical encoding first**: Converts all categorical variables to numerical format for subsequent operations
2. **Outlier treatment before scaling**: Prevents extreme values from skewing scaling parameters
3. **Scaling before imputation**: Ensures KNN distance calculations aren't dominated by unscaled features
4. **Column dropping before imputation**: Removes unnecessary columns to improve imputation efficiency
5. **Imputation last**: Uses all preprocessed features to make informed estimates for missing values

## Performance Considerations

- **Robust scaling over standard scaling**: Better handles the presence of outliers in student behavior data
- **Outer fence Tukey method**: Preserves legitimate variation while removing clear data errors
- **KNN imputation**: Leverages correlations between study habits and performance for better missing value estimates
- **Ordinal encoding**: Preserves meaningful relationships in categorical variables (education levels, quality ratings)

## Feature Engineering Notes

- **Exercise frequency and mental health rating**: Left unscaled as they're already on reasonable integer scales
- **Time-based features**: Study, social media, and Netflix hours capture time allocation patterns
- **Socioeconomic indicators**: Parental education and internet quality reflect student resources
- **Health indicators**: Sleep hours and diet quality capture physical wellness factors

## Expected Model Performance Impact

- **Strong predictors**: Study hours, attendance percentage, sleep hours, parental education level
- **Behavioral indicators**: Social media and Netflix hours may show negative correlation with performance
- **Health factors**: Diet quality and sleep hours should positively correlate with academic success
- **Resource access**: Internet quality and parental education reflect learning environment quality
