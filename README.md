# Math 2306: ODE Early-Semester Grade Analysis and Course Success Prediction

## Data description
Early-semester grade data collected over (at least) five semesters of Math 2306 students at Kennesaw State University.
Data include (for each student):
- Semester
- Four homework sets
- Three quizzes
- One midterm

From these data, the following are calculated:
- Homework average
- Quiz average
- Weighted course average, as follows: 10*(HW average) + 25*(Quiz average) + 65*(Midterm 1 score)

The final course letter grade for each student is stored as a target variable for prediction: students who earn an A, B, or C are labeled with a 1 (for passed) and students who earn a D or F are labeled with a 0 (for failed). 

**Important**: Dataset only includes grade data from students who finished the semester. Students who withdrew from the course are excluded.

## Exploratory Data Analysis

Correlation:
  
| Feature  | VIF      |
| -------- | -------  |
| HW Avg   |15.618070 |
| Midterm1 | 20.651488|
| Quiz Avg | 16.926606|

![image](https://github.com/user-attachments/assets/80fca629-0bca-4b48-aa90-a26cd627f3e2)
