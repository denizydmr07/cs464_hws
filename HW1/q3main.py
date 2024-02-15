import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
'''
I added below lines to avoid 
" RuntimeWarning: divide by zero encountered in log "
'''
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def main(visualize_plots=False):
    prefix = ""
    x_train = pd.read_csv(prefix+'X_train.csv', sep=" ")
    x_test = pd.read_csv(prefix+'X_test.csv', sep=" ")

    # labels can be loaded as series
    y_train = pd.read_csv(prefix+'y_train.csv', sep=" ", header=None)
    y_test = pd.read_csv(prefix+'y_test.csv', sep=" ", header=None)

    # convert to numpy array
    y_train = y_train.values
    y_test = y_test.values

    # squeeze the arrays
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    # label names
    class_labels = np.array([ "Bussiness", "Entertainment", "Politics", "Sport", "Tech"])

    y_train_unique, y_train_counts = np.unique(y_train, return_counts=True)
    y_test_unique, y_test_counts = np.unique(y_test, return_counts=True)

    if visualize_plots:
        print("Question 3.1.1")
        print("-"*50)

        fig, axes = plt.subplots(1,2, figsize=(10,5))
        fig.patch.set_facecolor("white")
        axes[0].pie(y_train_counts, labels=class_labels, autopct='%1.1f%%')
        axes[0].set_title("Percantage of each category in y_train")
        axes[1].pie(y_test_counts, labels=class_labels, autopct='%1.1f%%')
        axes[1].set_title("Percantage of each category in y_test")
        plt.show()
        print("-"*50)


    print("Question 3.1.2")
    print("-"*50)

    print("Prior probability for each class in the train set:\n")
    class_probs = y_train_counts/len(y_train)
    for i, prob in enumerate(class_probs):
        print("P({0}) = {1:.3f}"
            .format(class_labels[i], prob))
        
    print("\nPrior probability for each class in the test set:\n")
    test_class_probs = y_test_counts/len(y_test)
    for i, prob in enumerate(test_class_probs):
        print("P({0}) = {1:.3f}"
            .format(class_labels[i], prob))
        
    print("\nPrior probability for each class in the all data:\n")
    all_class_probs = (y_train_counts + y_test_counts)/(len(y_train) + len(y_test))
    for i, prob in enumerate(all_class_probs):
        print("P({0}) = {1:.3f}"
            .format(class_labels[i], prob))
        
    print("-"*50)

    # calculate "alien" frequency in "Tech" class in train set
    # get the tech indices
    tech_indices = np.where(y_train == 4)[0]

    # get the alien count in Tech class
    alien_count = sum(x_train.iloc[tech_indices]["alien"])
    # get the thunder count in Tech class
    thunder_count = sum(x_train.iloc[tech_indices]["thunder"])

    # get the alien frequency in Tech class
    alien_freq = alien_count / len(tech_indices)

    # get the thunder frequency in Tech class
    thunder_freq = thunder_count / len(tech_indices)


    print("Question 3.1.4")
    print("-"*50)

    print("Alien count in Tech class in train set: {0}".format(alien_count))
    print("Thunder count in Tech class in train set: {0}".format(thunder_count))
    print("Alien log ratio in Tech class in train set: {0:.4f}".format(np.log(alien_freq)))
    print("Thunder log ratio in Tech class in train set: {0:.4f}".format(np.log(thunder_freq)))

    print("-"*50)


    class MultinomialNaiveBayesClassifier():
        ''' 
        self.fitted: boolean, default False, indticating if the classifier is fitted or not
        self.class_labels: array of class labels, Business, Entertainment, Politics, Sport, Tech
        self.alpha: float, smoothing parameter
        self.y_train_unique: array, unique values of y_train
        self.y_train_counts: array, counts of unique values of y_train
        self.class_probs: array, prior probabilities of each class
        self.num_classes: int, number of classes
        self.probabilities: pandas dataframe, shape (num_classes, n_features), probability of each word in each class
        '''
        def __init__(self, class_labels):
            '''
            Initializes the classifier with given class labels

            Parameters:
                class_labels: array, labels of each class
            '''
            self.fitted = False
            self.class_labels = class_labels

        def fit(self, X, y, alpha=0):
            '''
            Gets a sample from the data (or all of the data) and corresponding labels 
            and calculates the probabilities of each word in each class.

            Parameters:
                X: pandas dataframe, shape (n_samples, n_features), features
                y: array, shape (n_samples,), labels
                alpha: float, smoothing parameter, default 0
            '''
            # if alpha is not given, set it to 0
            self.alpha = alpha
            # getting unique values of y_train and their counts
            self.y_train_unique, self.y_train_counts = np.unique(y, return_counts=True)
            # getting the prior probabilities of each class
            self.class_probs = self.y_train_counts/len(y)
            # number of classes
            self.num_classes = len(self.class_labels)

            # initializing the probabilities matrix
            self.probabilities = np.zeros((self.num_classes, X.shape[1]))
            for i in range(self.num_classes):
                # getting all rows with class i
                indicies = np.where(y == i)[0]
            
                # getting the occurences of each word in the class i, INCLUDING multiple occurences
                occurences = X.iloc[indicies].sum(axis=0)

                # calculating the probability of each word in the class i
                occurences = (occurences + self.alpha) / (sum(occurences) + self.alpha * X.shape[1])

                # adding the probabilities to the dataframe
                self.probabilities[i] = occurences.values

            # converting the probabilities to a dataframe
            self.probabilities = pd.DataFrame(self.probabilities, columns=X.columns, index=self.class_labels)

            #print(self.probabilities.unstack().unique()[self.probabilities.unstack().unique() < 0])

            # setting the fitted attribute to True
            self.fitted = True

        def classify(self, X):
            '''
            Classifies the given data with the fitted classifier.

            Parameters:
                X: pandas dataframe, shape (n_samples, n_features), features 

            Returns:
                array, shape (n_samples,), predicted labels
            '''

            # if the classifier is not fitted, raise an exception
            if not self.fitted:
                raise Exception("Classifier not fitted yet.")

            # calculating the log of probabilities using linear algebra instead of loops
            # using linear algebra is much faster
            # corresponds to eq. 3.6 in the HW
            return np.argmax(((X @ np.log(self.probabilities.T)) + np.log(self.class_probs)), axis=1 )       


    # initializing the classifier
    mnbc = MultinomialNaiveBayesClassifier(class_labels)
    # fitting the classifier
    mnbc.fit(x_train, y_train, 0)
    # predicting the labels
    y_pred_alpha_0 = mnbc.classify(x_test)

    def calculate_accuracy(y_true, y_pred):
        '''
        Calculates the accuracy.

        Parameters:
            y_true: array, shape (n_samples,), true labels
            y_pred: array, shape (n_samples,), predicted labels

        Returns:
            float, accuracy
        '''
        return np.mean(y_true == y_pred)


    def confusion_matrix(y_true, y_pred, class_labels):
        '''
        Calculates the confusion matrix.

        Parameters:
            y_true: array, shape (n_samples,), true labels
            y_pred: array, shape (n_samples,), predicted labels
            class_labels: array, labels of each class

        Returns:
            array, shape (num_classes, num_classes), confusion matrix
        '''
        num_classes = len(class_labels)
        cm = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                cm[i,j] = np.sum((y_true == i) & (y_pred == j))
        return cm


    def visualize_confusion_matrix(y_true, y_pred, class_labels):
        '''
        Visualizes the confusion matrix.

        Parameters:
            y_true: array, shape (n_samples,), true labels
            y_pred: array, shape (n_samples,), predicted labels
            class_labels: array, labels of each class
        '''

        # calculate the confusion matrix
        cm = confusion_matrix(y_true, y_pred, class_labels)
        _, ax = plt.subplots(figsize=(10,10))
        # plot the confusion matrix
        sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = "Reds", ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    print("Question 3.2")
    print("-"*50)

    print("Accuracy on test set when alpha is 0 (Multinomial): {0:.3f}".format(calculate_accuracy(y_test, y_pred_alpha_0)))
    if visualize_plots:
        print("Confusion matrix when alpha is 0 (Multinomial):")
        visualize_confusion_matrix(y_test, y_pred_alpha_0, class_labels)

    print("-"*50)

    # fitting the classifier with alpha = 1
    mnbc.fit(x_train, y_train, 1)
    # predicting the labels
    y_pred_alpha_1 = mnbc.classify(x_test)


    print("Question 3.3")
    print("-"*50)

    print("Accuracy on test set when alpha is 1 (Multinomial): {0:.3f}".format(calculate_accuracy(y_test, y_pred_alpha_1)))
    if visualize_plots:
        print("Confusion matrix when alpha is 1 (Multinomial):")
        visualize_confusion_matrix(y_test, y_pred_alpha_1, class_labels)

    print("-"*50)

    class BernoulliNaiveBayesClassifier():
        '''
        self.fitted: boolean, default False, indticating if the classifier is fitted or not
        self.class_labels: array of class labels, Business, Entertainment, Politics, Sport, Tech
        self.alpha: float, smoothing parameter
        self.y_train_unique: array, unique values of y_train
        self.y_train_counts: array, counts of unique values of y_train
        self.class_probs: array, prior probabilities of each class
        self.num_classes: int, number of classes
        self.probabilities: pandas dataframe, shape (num_classes, n_features), probability of each word in each class
        '''
        def __init__(self, class_labels):
            '''
            Initializes the classifier with given class labels
            
            Parameters:
                class_labels: array, labels of each class
            '''
            self.fitted = False
            self.class_labels = class_labels

        def fit(self, X, y, alpha):
            '''
            Gets a sample from the data (or all of the data) and corresponding labels 
            and calculates the probabilities of each word in each class.

            Parameters:
                X: pandas dataframe, shape (n_samples, n_features), features
                y: array, shape (n_samples,), labels
                alpha: float, smoothing parameter, default 0
            '''
            # if alpha is not given, set it to 0
            self.alpha = alpha
            # getting unique values of y_train and their counts
            self.y_train_unique, self.y_train_counts = np.unique(y, return_counts=True)
            # getting the prior probabilities of each class
            self.class_probs = self.y_train_counts/len(y)
            # number of classes
            self.num_classes = len(self.class_labels)

            # initializing the probabilities matrix
            self.probabilities = np.zeros((self.num_classes, X.shape[1]))

            for i in range(self.num_classes):
                # getting all rows with class i
                indicies = np.where(y == i)[0]

                # getting the occurences of each word in the class i, EXCLUDING multiple occurences
                occurences = (X.iloc[indicies] >= 1).sum(axis=0)

                # calculating the probability of each word in the class i
                occurences = (occurences + self.alpha) / (y_train_counts[i] + 2 * self.alpha)

                # adding the probabilities to the dataframe
                self.probabilities[i] = occurences.values

            # converting the probabilities to a dataframe
            self.probabilities = pd.DataFrame(self.probabilities, columns=X.columns, index=self.class_labels)

            # setting the fitted attribute to True
            self.fitted = True 

        def classify(self, X):
            '''
            Classifies the given data with the fitted classifier.

            Parameters:
                X: pandas dataframe, shape (n_samples, n_features), features 

            Returns:
                array, shape (n_samples,), predicted labels
            '''
            # if the classifier is not fitted, raise an exception
            if not self.fitted:
                raise Exception("Classifier not fitted yet.")
            
            # calculating the log of probabilities using linear algebra instead of loops
            # using linear algebra is much faster than using loops
            # corresponds to eq. 3.7 in the HW
            return np.argmax((X >= 1) @ np.log(self.probabilities.T) + (1 - (X >= 1)) @ np.log(1 - self.probabilities.T) + np.log(self.class_probs), axis=1)
                
    # initializing the classifier
    bnbc = BernoulliNaiveBayesClassifier(class_labels)
    # fitting the classifier with alpha = 1
    bnbc.fit(x_train, y_train, 1)
    # predicting the labels
    y_pred_alpha_1_bernoulli = bnbc.classify(x_test)


    print("Question 3.4")
    print("-"*50)

    print("Accuracy on test set when alpha is 1 (Bernoulli): {0:.3f}".format(calculate_accuracy(y_test, y_pred_alpha_1_bernoulli)))
    if visualize_plots:
        print("Confusion matrix when alpha is 1 (Bernoulli):")
        visualize_confusion_matrix(y_test, y_pred_alpha_1_bernoulli, class_labels)

    print("-"*50)

if __name__ == "__main__":
    main()