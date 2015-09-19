 slidenumbers: true
 autoscale:true

![inline, left, 30%](/Users/rahul/Dropbox/Public/images/logo1.png)

 Boston, Sep 19th, 2015

#[fit] ACTING ON DATA

###Rahul Dave (rahuldave@gmail.com)

###@rahuldave

![30%, Left, inline](/Users/rahul/Desktop/lxpriorlogo.pdf)    

 ![40%, right, inline](/Users/rahul/Desktop/iacslogo.png)

---

![](/Users/rahul/Desktop/working-noir.jpeg)

![60%, inline](/Users/rahul/Desktop/lxpriorlogo.pdf)

###Data Science.
###Simulation.
###Software.

###Social Good.
###Interesting Problems.
---

![70%, inline](/Users/rahul/Desktop/iacslogo.png)

machine learning, complex systems, stochastic methods, viz, extreme computing

###DEGREE PROGRAMS:
- Master’s of science– one year
- Master’s of engineering – two year with thesis/research project


---

#[fit]CLASSIFICATION

- will a customer churn?
- is this a check? For how much?
- a man or a woman?
- will this customer buy?
- do you have cancer?
- is this spam?
- whose picture is this?
- what is this text about?[^j]

![fit, left](/Users/rahul/Desktop/bookimages/onelinesplit.pdf)

[^j]:image from code in http://bit.ly/1Azg29G

---
#[fit]REGRESSION

- how many dollars will you spend?
- what is your creditworthiness
- how many people will vote for Bernie t days before election
- use to predict probabilities for classification
- causal modeling in econometrics

![fit, right](/Users/rahul/Desktop/bookimages/linreg.pdf)

---

#[fit]From Bayesian Reasoning and Machine Learning, David Barber:

"A father decides to teach his young son what a sports car is. Finding it difficult to explain in words, he decides to give some examples. They stand on a motorway bridge and ... the father cries out ‘that’s a sports car!’ when a sports car passes by. After ten minutes, the father asks his son if he’s understood what a sports car is. The son says, ‘sure, it’s easy’. An old red VW Beetle passes by, and the son shouts – ‘that’s a sports car!’. Dejected, the father asks – ‘why do you say that?’. ‘Because all sports cars are red!’, replies the son."

---

30 points of data. Which fit is better? Line in $$\cal{H_1}$$ or curve in $$\cal{H_{20}}$$?

![inline, fit](/Users/rahul/Desktop/presentationimages/linearfit.png)![inline, fit](/Users/rahul/Desktop/presentationimages/20thorderfit.png)

---

![fit, left](/Users/rahul/Desktop/bookimages/linreg.pdf)

#[fit]What does it mean to FIT?

Minimize distance from the line?

$$R_{\cal{D}}(h_1(x)) = \frac{1}{N} \sum_{y_i \in \cal{D}} (y_i - h_1(x_i))^2 $$

Minimize squared distance from the line.

$$ g_1(x) = \arg\min_{h_1(x) \in \cal{H}} R_{\cal{D}}(h_1(x)).$$

##[fit]Get intercept $$w_0$$ and slope $$w_1$$.

---

#EMPIRICAL RISK MINIMIZATION

The sample must be representative of the population!

![fit, left](/Users/rahul/Desktop/presentationimages/inputdistribution.png)

$$A : R_{\cal{D}}(g) \,\,smallest\,on\,\cal{H}$$
$$B : R_{out \,of \,sample} (g) \approx R_{\cal{D}}(g)$$


A: Empirical risk estimates out of sample risk.
B: Thus the out of sample risk is also small.

---
# THE REAL WORLD HAS NOISE

Which fit is better now?
                                              The line or the curve?

![fit, inline](/Users/rahul/Desktop/presentationimages/fitswithnoise.png)

---

![filtered](/Users/rahul/Desktop/bookimages/bias.pdf)

#[fit] DETERMINISTIC NOISE (Bias) vs STOCHASTIC NOISE

![inline, fit, left](/Users/rahul/Desktop/presentationimages/errorwithnoise.png)
![inline, fit, right](/Users/rahul/Desktop/presentationimages/errorwithnonoise.png)

---

#UNDERFITTING (Bias)
#vs OVERFITTING (Variance)

![inline, fit](/Users/rahul/Desktop/presentationimages/varianceinfits.png)

---
![](/Users/rahul/Desktop/presentationimages/wiki.png)


#[fit]If you are having problems in machine learning the reason is almost always[^-]

#[fit]OVERFITTING

[^-]: Background from http://commons.wikimedia.org/wiki/File:Overfitting.svg

---
DATA SIZE MATTERS: straight line fits to a sine curve

![inline, fit](/Users/rahul/Desktop/presentationimages/datasizematterssine.png)

Corollary: Must fit simpler models to less data!

---

#HOW DO WE LEARN?

![inline](/Users/rahul/Desktop/bookimages/train-test.pdf)

![right, fit](/Users/rahul/Desktop/presentationimages/testandtrainingpoints.png)

---

![left, fit](/Users/rahul/Desktop/bookimages/complexity-error-plot.pdf)

#BALANCE THE COMPLEXITY

![inline](/Users/rahul/Desktop/presentationimages/trainingfit.png)

---

![right, fit](/Users/rahul/Desktop/bookimages/train-validate-test3.pdf)

#[fit]VALIDATION

- train-test not enough as we *fit* for $$d$$ on test set and contaminate it
- thus do train-validate-test

![inline](/Users/rahul/Desktop/bookimages/train-validate-test.pdf)


---

#[fit]CROSS-VALIDATION

![inline](/Users/rahul/Desktop/bookimages/train-cv.pdf)

![right, fit](/Users/rahul/Desktop/bookimages/train-cv3.pdf)

---


![original, right, fit](/Users/rahul/Desktop/bookimages/complexity-error-reg.pdf)

#[fit]REGULARIZATION

Keep higher a-priori complexity and impose a

#complexity penalty

on risk instead, to choose a SUBSET of $$\cal{H}_{big}$$. We'll make the coefficients small:

$$\sum_{i=0}^j a_i^2 < C.$$

---

![fit](/Users/rahul/Desktop/presentationimages/regularizedsine.png)

---

![original, left, fit](/Users/rahul/Desktop/presentationimages/regwithcoeff.png)

#[fit]REGULARIZATION
$$\cal{R}(h_j) =  \sum_{y_i \in \cal{D}} (y_i - h_j(x_i))^2 +\alpha \sum_{i=0}^j a_i^2.$$

As we increase $$\alpha$$, coefficients go towards 0.

Lasso uses $$\alpha \sum_{i=0}^j |a_i|,$$ sets coefficients to exactly 0.

Thus regularization automates:

       FEATURE ENGINEERING

---

#[fit]CLASSIFICATION

#[fit]BY LINEAR SEPARATION

#Which line?

- Different Algorithms, different lines.

- SVM uses max-margin[^j]

![fit, right](/Users/rahul/Desktop/bookimages/linsep.pdf)

---

#DISCRIMINATIVE CLASSIFIER
$$P(y|x): P(male | height, weight)$$

![inline, fit](/Users/rahul/Desktop/presentationimages/logis.png)![inline, fit](/Users/rahul/Desktop/presentationimages/probalogis.png)

##VS. DISCRIMINANT

---

#GENERATIVE CLASSIFIER
$$P(y|x) \propto P(x|y)P(x): P(height, weight | male) \times P(male)$$

![inline, fit](/Users/rahul/Desktop/presentationimages/lda.png)![inline, fit](/Users/rahul/Desktop/presentationimages/probalda.png)


---

#[fit]The virtual ATM[^:]

![left](/Users/rahul/Desktop/presentationimages/check.png)
![right](/Users/rahul/Desktop/presentationimages/dollar.png)

[^:]: Inspired by http://blog.yhathq.com/posts/image-classification-in-Python.html

---

#[fit] PCA[^ ]     
#[fit]unsupervised dim reduction from 332x137x3  to 50

![100%,  left](/Users/rahul/Desktop/presentationimages/pcanim.gif)

![inline](/Users/rahul/Desktop/presentationimages/pca0.png)
![inline](/Users/rahul/Desktop/presentationimages/pca1.png)

[^ ]: Diagram from http://stats.stackexchange.com/a/140579

---

#kNN

![inline](/Users/rahul/Desktop/bookimages/knn1.pdf)

![right, fit](/Users/rahul/Desktop/presentationimages/twoprincipalsatm.png)

---
#[fit] BIAS and VARIANCE in KNN
![left, 99%](/Users/rahul/Desktop/bookimages/knn2.pdf)

![inline](/Users/rahul/Desktop/presentationimages/k1onatm.png)

![inline](/Users/rahul/Desktop/presentationimages/k40onatm.png)


---
#kNN
#CROSS-VALIDATED


![left, fit](/Users/rahul/Desktop/presentationimages/cvalforknnatm.png)

![inline](/Users/rahul/Desktop/presentationimages/cvalsonatm.png)

---

#[fit]ENSEMBLE LEARNING

![inline](/Users/rahul/Desktop/presentationimages/randforest.png)

#Combine multiple classifiers and vote.

Egs: Decision Trees $$\to$$ random forest, bagging, boosting

---

#EVALUATING CLASSIFIERS

![left, fit](/Users/rahul/Desktop/bookimages/confusionmatrix.pdf)

![inline](/Users/rahul/Desktop/presentationimages/cvalsonatm.png)

```
confusion_matrix(bestcv.predict(Xtest), ytest)

[[11, 0],
 [3, 4]]
```
checks=blue=1, dollars=red=0

---

#[fit]CLASSIFIER PROBABILITIES

- classifiers output rankings or probabilities
- ought to be well callibrated, or, atleast, similarly ordered

![right, fit](/Users/rahul/Desktop/presentationimages/nbcallib.png)

![inline](/Users/rahul/Desktop/presentationimages/calibrationhw.png)

---

#CLASSIFICATION RISK

- $$R_{g,\cal{D}}(x) = P(y_1 | x) \ell(g,y_1) + P(y_0 | x) \ell(g,y_0) $$
- The usual loss is the 1-0 loss $$\ell = \mathbb{1}_{g \ne y}$$.
- Thus, $$R_{g=y_1}(x) = P(y_0 |x)$$ and $$R_{g=y_0}(x) = P(y_1 |x)$$

       CHOOSE CLASS WITH LOWEST RISK

1 if $$R_1 \le R_0 \implies$$ 1 if $$P(0 | x) \le P(1 |x)$$.

       **choose 1 if $$P(1|x) \ge 0.5$$ ! Intuitive!**

---

![](/Users/rahul/Desktop/presentationimages/brcancer.png)

#ASYMMETRIC RISK[^^]

want no checks misclassified as dollars, i.e., no false negatives: $$\ell_{10} \ne \ell_{01}$$.

Start with $$R_{g}(x) = P(1 | x) \ell(g,1) + P(0 | x) \ell(g,0) $$

$$R_1 = l_{11} p_1 + l_{10} p_0 = l_{10}p_0$$
$$R_0 = l_{01} p_1 + l_{00} p_0 = l_{01}p_1$$

**choose 1 if $$R_1 < R_0$$**

[^^]:image of breast carcinoma from http://bit.ly/1QgqhBw

---

#ASYMMETRIC RISK

![right, fit](/Users/rahul/Desktop/presentationimages/asymmriskknn.png)

i.e. $$r p_0 < p_1$$ where

$$r = \frac{l_{10}}{l_{01}} = \frac{l_{FP}}{l_{FN}}$$

Or $$P_1 > t$$ where $$t = \frac{r}{1+r}$$


```
confusion_matrix(ytest, bestcv2, r=10, Xtest)
[[5, 4],
 [0, 9]]
```
---

#[fit]ROC SPACE[^+]


$$TPR = \frac{TP}{OP} = \frac{TP}{TP+FN}.$$

$$FPR = \frac{FP}{ON} = \frac{FP}{FP+TN}$$

![left, fit](/Users/rahul/Desktop/presentationimages/ROCspace.png)

![inline](/Users/rahul/Desktop/bookimages/confusionmatrix.pdf)

[^+]: this+next fig: Data Science for Business, Foster et. al.

---

![fill, Original](/Users/rahul/Desktop/presentationimages/howtoroc.png)

![inline, right, 70%, filtered](/Users/rahul/Desktop/presentationimages/roc-curve.png)

#[fit]ROC CURVE


---

#ROC CURVE

- Rank test set by prob/score from highest to lowest
- At beginning no +ives
- Keep moving threshold
- confusion matrix at each threshold

![fit, original](/Users/rahul/Desktop/presentationimages/hwroclogistic.png)

---

#[fit]COMPARING CLASSIFERS

Telecom customer Churn data set from @YhatHQ[^<]

![inline](/Users/rahul/Desktop/presentationimages/churn.png)

[^<]: http://blog.yhathq.com/posts/predicting-customer-churn-with-sklearn.html

---

#ROC curves

![fit, original](/Users/rahul/Desktop/presentationimages/roccompare.png)

---

#ASYMMETRIC CLASSES

- A has large FP[^#]
- B has large FN.
- On asymmetric data sets, A will do very bad.
- But is it so?



![fit, left](/Users/rahul/Desktop/bookimages/abmodeldiag.png)


---

#EXPECTED VALUE FORMALISM

Can be used for risk or profit/utility (negative risk)

$$EP = p(1,1) \ell_{11} + p(0,1) \ell_{10} + p(0,0) \ell_{00} + p(1,0) \ell_{01}$$


$$EP = p_a(1) [TPR\,\ell_{11} + (1-TPR) \ell_{10}]$$
         $$ + p_a(0)[(1-FPR) \ell_{00} + FPR\,\ell_{01}]$$

Fraction of test set pred to be positive $$x=PP/N$$:

$$x = (TP+FP)/N = TPR\,p_o(1) + FPR\,p_o(0)$$

---

#ASYMMETRIC CLASSES


$$r = \frac{l_{FP}}{l_{FN}}$$

Equicost lines with slope

$$\frac{r\,p(0)}{p(1)} = \frac{r\,p(-)}{r\,p(+)}$$

Small $$r$$ penalizes FN.

Churn and Cancer u dont want FN: an uncaught churner or cancer patient (P=churn/cancer)

![fit, right](/Users/rahul/Dropbox/Public/images/bookimages/asyroc.png)

---

![fit, original](/Users/rahul/Desktop/presentationimages/profitcompare.png)

#Profit curve

- Rank test set by prob/score from highest to lowest
- Calculate the expected profit/utility for each confusion matrix ($$U$$)
- Calculate fraction of test set predicted as positive ($$x$$)
- plot $$U$$ against $$x$$

---

#Finite budget[^#]

![fit, right](/Users/rahul/Desktop/presentationimages/pcurves.png)


- 100,000 customers, 40,000 budget, 5$ per customer
- we can target 8000 customers
- thus target top 8%
- classifier 1 does better there, even though classifier 2 makes max profit

[^#]: figure from Data Science for Business, Foster et. al.

---

#[fit]WHERE TO GO FROM HERE?

- Follow @YhatHQ, @kdnuggets etc
- Read Provost, Foster; Fawcett, Tom. Data Science for Business: What you need to know about data mining and data-analytic thinking. O'Reilly Media.
- check out Harvard's cs109: cs109.org
- Ask lots of questions of your data science team
- Follow your intuition

#THANKS!!

###@rahuldave
