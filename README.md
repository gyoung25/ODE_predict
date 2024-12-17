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

### Correlation

Let's look at the feature correlation matrix:

|          |       HW1 |       HW2 |       HW3 |        HW4 |   HW Avg |     Quiz1 |      Quiz2 |     Quiz3 |   Quiz Avg |   Midterm1 |     Pass |
|:---------|----------:|----------:|----------:|-----------:|---------:|----------:|-----------:|----------:|-----------:|-----------:|---------:|
| HW1      | 1         | 0.253373  | 0.361497  |  0.394416  | 0.696191 | 0.189396  |  0.191786  | 0.200634  |  0.263491  |  0.0314292 | 0.12574  |
| HW2      | 0.253373  | 1         | 0.154089  |  0.487146  | 0.697526 | 0.0874593 |  0.128916  | 0.156878  |  0.166044  |  0.0442056 | 0.216727 |
| HW3      | 0.361497  | 0.154089  | 1         |  0.282241  | 0.629549 | 0.17412   |  0.0528499 | 0.0939185 |  0.149261  |  0.110683  | 0.131569 |
| HW4      | 0.394416  | 0.487146  | 0.282241  |  1         | 0.779491 | 0.0970247 | -0.0182701 | 0.057985  |  0.0636405 |  0.0835274 | 0.169171 |
| HW Avg   | 0.696191  | 0.697526  | 0.629549  |  0.779491  | 1        | 0.191931  |  0.124251  | 0.179998  |  0.225561  |  0.096318  | 0.231919 |
| Quiz1    | 0.189396  | 0.0874593 | 0.17412   |  0.0970247 | 0.191931 | 1         |  0.445091  | 0.143003  |  0.761957  |  0.326459  | 0.315997 |
| Quiz2    | 0.191786  | 0.128916  | 0.0528499 | -0.0182701 | 0.124251 | 0.445091  |  1         | 0.313793  |  0.806652  |  0.365008  | 0.364305 |
| Quiz3    | 0.200634  | 0.156878  | 0.0939185 |  0.057985  | 0.179998 | 0.143003  |  0.313793  | 1         |  0.61866   |  0.482089  | 0.360789 |
| Quiz Avg | 0.263491  | 0.166044  | 0.149261  |  0.0636405 | 0.225561 | 0.761957  |  0.806652  | 0.61866   |  1         |  0.524909  | 0.470391 |
| Midterm1 | 0.0314292 | 0.0442056 | 0.110683  |  0.0835274 | 0.096318 | 0.326459  |  0.365008  | 0.482089  |  0.524909  |  1         | 0.661899 |
| Pass     | 0.12574   | 0.216727  | 0.131569  |  0.169171  | 0.231919 | 0.315997  |  0.364305  | 0.360789  |  0.470391  |  0.661899  | 1        |

From the above correlation matrix, it seems HW Avg, Quiz Avg, and Midterm1 correlate most with the target variable Pass.

Let's limit our search to these three explanatory variables. Next, we'll calculate the [variance inflation factor](https://en.wikipedia.org/wiki/Variance_inflation_factor) (VIF) of there variables to determine if we need all of them.

| Feature  | VIF      |
| -------- | -------  |
| HW Avg   |15.618070 |
| Midterm1 | 20.651488|
| Quiz Avg | 16.926606|

![image](https://github.com/user-attachments/assets/80fca629-0bca-4b48-aa90-a26cd627f3e2)
