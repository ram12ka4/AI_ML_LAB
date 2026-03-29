The problem of overfitting

Underfitting and overfitting are just about how simple or how complex your model is, and whether it makes good predictions on new data.

1. The core idea
Underfitting (high bias): Model is too simple, cannot even fit the training data pattern properly.

Overfitting (high variance): Model is too complex, fits training data almost perfectly, but fails badly on new data.

Just right: Model fits training data reasonably and also works well on new, unseen data (good generalization).

2. House price example (regression)
Imagine 5 houses with (size → price):

Underfitting / high bias

Use a straight line (simple linear model)

But true relationship is curved (prices flatten for big houses)

Line misses clear pattern → bad on training and bad on new data.

Just right

Use a quadratic model (features: x,x2x,x2 )

Curve follows the overall pattern: increasing then flattening

Not perfect on all points, but captures real trend → good on new houses.

Overfitting / high variance

Use a 4th-degree polynomial (features: 
x,
x
2
,
x
3
,
x
4
x,x 
2
 ,x 
3
 ,x 
4
 )

Curve goes exactly through all 5 points → training cost = 0

But it wiggles crazily between points; may predict that a bigger house is cheaper than smaller ones

Very good on training data, bad on new data.

Too few features → underfit, too many/too flexible features → overfit.

3. Classification example (logistic regression)
Features:

x
1
x 
1
​
 : tumor size

x
2
x 
2
​
 : age
Labels: malignant (×) vs benign (○)

Underfitting / high bias

Model: simple logistic regression, 
z
=
w
1
x
1
+
w
2
x
2
+
b
z=w 
1
​
 x 
1
​
 +w 
2
​
 x 
2
​
 +b

Decision boundary: straight line

Roughly separates classes but misses structure → not great on training or new data.

Just right

Add quadratic features like 
x
1
2
,
x
2
2
,
x
1
x
2
x 
1
2
​
 ,x 
2
2
​
 ,x 
1
​
 x 
2
​
 

Decision boundary becomes curved (ellipse-like)

Does not classify all training points perfectly, but matches real cluster shape → generalizes well.

Overfitting / high variance

Add many higher-order polynomial features

Model finds a weird, twisty decision boundary that perfectly separates all training points

But that boundary is unrealistic; small change in data would change it a lot → poor performance on new patients.

4. Bias vs variance wording
High bias ≈ underfitting

Strong assumption like “relationship must be purely linear”

Model ignores evidence in data that reality is more complex.

High variance ≈ overfitting

Model bends too much to match every training point

Small change in training data → big change in predictions.

People often use the pairs almost interchangeably:

underfitting ↔ high bias

overfitting ↔ high variance

5. Goldilocks analogy
Too simple (underfit / high bias) → “porridge too cold”

Too complex (overfit / high variance) → “porridge too hot”

Just right → good balance: fits training data reasonably, and generalizes to new data.


Here he’s answering: “Once I know my model is overfitting, what can I do?”
There are 3 main tools:

1. Get more training data
Overfitting = model learned too much detail / noise from a small dataset.

If you add more examples, the model is forced to learn the real pattern instead of memorizing individual points.

In the house example: more (size, price) points → the high‑degree polynomial will become less wiggly and smoother.

Problem: Sometimes more data is not available (only limited houses, medical cases, etc).

2. Use fewer or simpler features (Feature Selection)
Two cases:

Polynomial features case

Features: 
x
,
x
2
,
x
3
,
x
4
,
…
x,x 
2
 ,x 
3
 ,x 
4
 ,…

Too many powers → very flexible curve → overfitting.

Fix: remove higher powers (e.g., keep only 
x
,
x
2
x,x 
2
 ). That reduces model complexity.

Many real-world features case
Example features for house price:

size, number of bedrooms, floors, age, neighborhood income, distance to coffee shop, etc.

If you have 100 features but few training examples, model can overfit.

Fix: select only the most useful subset (e.g., size, bedrooms, age). This is called feature selection.

Tradeoff:

Removing features = throwing away some information.

Maybe all 100 are useful, so you might prefer not to drop them if you can control overfitting in another way (see next).

3. Regularization (shrink parameters, keep all features)
Idea:

Overfitting often happens when some weights 
w
j
w 
j
​
  become large, letting some features dominate and create weird curves/decision boundaries.

Removing a feature completely = setting its weight 
w
j
=
0
w 
j
​
 =0.

Regularization is a softer version: it pushes weights towards 0, but not necessarily exactly 0.

Effect:

You can still use many features (even high-order polynomials), but regularization keeps their influence small, so the model becomes smoother and less wiggly.

You get a curve/boundary that fits training data reasonably but not too perfectly, and generalizes better.

Conventions:

Usually we regularize only 
w
1
…
w
n
w 
1
​
 …w 
n
​
 , not 
b
b.

Regularizing or not regularizing 
b
b rarely changes performance much; most people skip regularization on 
b
b.

Regularization (L2 style) is used widely, especially in neural networks and other models, to reduce overfitting.

Summary of the 3 tools
More data – best if possible; reduces overfitting naturally.

Fewer / better-chosen features – manual or automatic feature selection.

Regularization – keep all features, but penalize large weights to control complexity.

If you want, I can next show you the exact cost function with regularization for linear regression and logistic regression, with a small numeric example.