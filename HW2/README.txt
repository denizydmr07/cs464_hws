1) I assumed that all the .gz files are under the folder named data in the same directory with the .py scripts.
If not, please change the first lines from the both scripts 
prefix = "" 
to the path of the folder where the .gz are located.

2) You can call the q1.py and q2.py scripts like "python q1.py" or "python3 q1.py"

3) I also added the notebook files for both scripts. You can check the scripts outputs from the notebooks if necesarry. 
The codes in the notebooks and the scripts are the same, I actually just imported scripts from the notebooks.
They only differ in the prefix variable, notebooks do not have it.
