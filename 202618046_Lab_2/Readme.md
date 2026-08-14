**Objectives covered in Part A**
-> In part A we learned how to create numpy arrays, using inbuilt numpy array function such as np.random.randint(), np.arange(), np.zeroes(),
also performed statistical calculation among numpy arrays using function like np.mean, np.max, np.min, np.std etc, explored array properties such as shape, type, dim,
len() with inbuilt numpy functions, learnt about 1D, 2D, 3D arrays, reshaping, flattening like techniques, performed Matrix Operations among numpy arrays.


**Objectives covered in Part B**
-> In part B we learnt in depth about data manipulation using pandas, loading the titanic dataset, viewing dataset and extracting basic statistical data, basic EDA.
-> slicing of data using loc and .loc, filtering of data using boolean condition, extracting column/group specific data using groupby and aggregation methods.
-> filled the null data with central tendency values such as mean, median and mode, also calculated IQR for filled data.
-> analysed the data for survival of passengers, determined the percentage of survivors based on their gender, income level, passenger class and derived the relationship among those features 
and derived these observations as explained below :

-> As per numerical observations we observed that overall females has higher possibility of survival as compared to male passengers, Approximately 74.20% of females survived, compared with only 18.89% of males. 
The passengers who belong to first class/upper class showed higher probability of survival than rest of passenger classes, here Pclass and Fare have a positive relationship indicating better facilities for high class, 
high income passengers, also lower pclass and survival showed negative relationship indicating lower suvival rate for class 3 passengers so it shows the passengers that gave more fare for trip had better survival rate.
The Age vs Fare scatter plot shows a wide range of fares for younger and older passengers. There is no simple linear relationship between age and fare, although some very high-fare observations are visible.correlation between SibSp and Parch is positive. 
those passengers travelling with siblings/spouses also tended to have parents/children with them, indicating a relationship between these two family-related variables.
