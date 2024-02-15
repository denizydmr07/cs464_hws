1) I assumed that all the .csv files are under the same folder with the q3main.py script. 
If not, please change the line 13:
prefix = "" 
to the path of the folder where the .csv files are located (e.g. .../datasets/).

2) You can call the q3main.py script like "python q3main.py" or "python3 q3main.py"

3) If you want to visualize the results (i.e. the confusion matrices, piechart),
please set the default parameter "visualize_plots" to True in line 12 
or set visualize_plots = True when calling the function "main()" in line 370.

4) Here is an example output of the q3main.py with visualize_plots = False:

Question 3.1.2
--------------------------------------------------
Prior probability for each class in the train set:

P(Bussiness) = 0.225
P(Entertainment) = 0.170
P(Politics) = 0.191
P(Sport) = 0.226
P(Tech) = 0.188

Prior probability for each class in the test set:

P(Bussiness) = 0.242
P(Entertainment) = 0.183
P(Politics) = 0.176
P(Sport) = 0.241
P(Tech) = 0.158

Prior probability for each class in the all data:

P(Bussiness) = 0.229
P(Entertainment) = 0.173
P(Politics) = 0.187
P(Sport) = 0.230
P(Tech) = 0.180
--------------------------------------------------
Question 3.1.4
--------------------------------------------------
Alien count in Tech class in train set: 3
Thunder count in Tech class in train set: 0
Alien log ratio in Tech class in train set: -4.6476
Thunder log ratio in Tech class in train set: -inf
--------------------------------------------------
Question 3.2
--------------------------------------------------
Accuracy on test set when alpha is 0 (Multinomial): 0.242
--------------------------------------------------
Question 3.3
--------------------------------------------------
Accuracy on test set when alpha is 1 (Multinomial): 0.977
--------------------------------------------------
Question 3.4
--------------------------------------------------
Accuracy on test set when alpha is 1 (Bernoulli): 0.966
--------------------------------------------------