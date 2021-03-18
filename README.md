# Logistic Regression

Code run through:
  We create a constructer that initiates the tolerance, iterations and leraning rate for the model.
  
**Define a method to read data**
  
    def datasetReader(self):
  
**A method to remove rows**
   This methid is to study the significance of outliers on the model. We use this method to remove the outliers and run the model to compare Recall, Accuracy and Precision with and without outliers. 
     
     def removeIndex(self, rows)
     train_df = train_df.drop(rows)
   
**Add Bias**
  Adding a column of ones to train bias
    
      def addX0(self, X):
        
        return np.column_stack([np.ones(X.shape[0], 1), X])
        
**Introduce Sigmod fn**
  This function scores the parameters, having a threshold on this score helps predict to which class the point belongs to.
  
      def sigmoid(self, z):
        
        sig = 1 / (1 + np.exp(-z))
        return sig
 **Calculate cost function**
    Goal is to minimize this through gradient descent
    
    def costFunction(self, X, y):
        
        sig = self.sigmoid(X.dot(self.w))  # QTX

        
        pred = np.log(np.ones(X.shape[0]) + np.exp(X.dot(self.w))) - X.dot(self.w).dot(y)
        cost = pred.sum()
        
        return cost
        
 **Gradient Descent**
    Take the derivative of cost function but do not equate it to zero
      
      def gradient(self, X, y):
        
        sig = self.sigmoid(X.dot(self.w))
        grad = (sig - y).dot(X)
        
        return grad
    
    def gradientDescent(self, X, y):
        
        costSequence = []
        
        prevCost = float('inf')
        
        toleranceCounter = 0
        
        for i in tqdm(range(self.maxIteration)):
            self.w = self.w - self.learningRate * self.gradient(X, y)
            currentCost = self.costFunction(X, y)
            diff = prevCost - currentCost
            prevCost = currentCost
            costSequence.append(currentCost)
            
            if diff < self.tolerance:
                toleranceCounter += 1
                print('The Model has stopped, no further improvement')
                break
                       
        self.plotCost(costSequence)
        return
        
   **Results**
    
    def predict(self, X):
        
        sig = self.sigmoid(X.dot(self.w))
        return np.around(sig)
        
 ![image](https://user-images.githubusercontent.com/75456477/111558819-e2b6a780-8765-11eb-8965-c514006fe282.png)

![Unknown](https://user-images.githubusercontent.com/75456477/111559802-90768600-8767-11eb-9d60-b0b51ec7d989.png)

![image](https://user-images.githubusercontent.com/75456477/111560403-c6683a00-8768-11eb-8fa2-71a358a2e5ab.png)

**Passing a plane as a decision boundary through multidimentional linearly seperable points**

![image](https://user-images.githubusercontent.com/75456477/111560436-d97b0a00-8768-11eb-9a90-e1ca9ebe4220.png)
