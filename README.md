# Pole_Sum_Net
An ANN classifier and regressor to predict the number of poles and find the pole parameters in a sum of real poles and cc pole pairs

Q: Where can it be applied?
A: Suppose you have a function f(z), where z is complex. 
Moreover, suppose that you know that f is a sum of N_r real poles and N_c complex conjugate pole pairs, where 1 <= N_r+N_c <=3, but you do not know N_r, N_c or the actual pole parameters (positions and coefficients). 
Finally, suppose that you have values of f on the real axis. For example, you may know the values of f(z) at z=0,1,2,3,..,99. 
Our classifier can tell you N_r and N_c and our regressor can tell you the actual pole parameters.
Note: actually, the classifier and regressor assume, that your data consists of two of those functions/ data curves, which shall only differ by the pole coefficients. 
I.e. one data sample shall consist of values f1(z1),f1(z2),...f1(z_n),f2(z1),f2(z2),...f2(zn), where z1,..zn are the used "grid" values on the real axis and f1 and f2 are the two pole functions with the same N_r, N_c and pole positions, but with differing pole coefficients.
If this is not the case, i.e. your data consists only of values of a single pole function f1, the code can easily be adapted.
(See also the "gap_equation" folder, where code is provided that can be used to generate Dyson-Schwinger equation results on which the ANNs can then be applied.)

Q: How does it work?
A: Find details in my Masters thesis.
https://unipub.uni-graz.at/obvugrhs/content/titleinfo/7708895?lang=en

Q: How do I use it?
A:
Classifier:
1.) Specify the wanted settings in parameters.py.
2.) Create training data by running create_data_classifier.py. Depending on your PC and the chosen settings, this may take days.
3.) Train the classifier by running train_pole_classifier.py. In line 149, you may need to specify which GPU/CPU you want to use (default: gpus=1).
4.) Apply the classifier to your data using test_data.py (which you will need to adjust to your data, see also 'Q: Where can it be applied?') and application_classifier.py.
Regressor:
1.) Set class_regressor in parameters.py to the value predicted by the classifier(s) and specify the remaining hyperparameters.
2.) Create data, train and apply the regressor with the corresponding files (the ones that contain the word "regressor" in their names).
SciPy:
1.) Use test_data.py and application_scipy_lm.py to apply SciPy's curve_fit function with the lm method onto your data.

Used programs:

Python (v. 3.9.7, https://www.python.org/), 
numpy (v. 1.20.3,https://numpy.org/), 
Pytorch (v. 1.10.1, https://pytorch.org/), 
Pytorch Lightning (v.1.2.3, https://www.pytorchlightning.ai/), 
sci-kit learn (v. 1.0.2, https://scikit-learn.org/), 
scipy (v. 1.7.1, https://scipy.org/), 
lmfit (v. 1.0.3, https://lmfit.github.io/lmfit-py/), 
matplotlib (v. 3.4.3, https://matplotlib.org/), 
joblib (v. 1.1.0, https://joblib.readthedocs.io), 
[wandb (v. 0.12.9, https://wandb.ai/)] 

How to read the regressor output:

Pole Class 0:   1 real pole,    0 cc pole pairs
[pos_re, coeff_re, coeff_re]

Pole Class 1:   0 real poles,   1 cc pole pair
[pos_re, pos_im, coeff_re, coeff_im, coeff_re, coeff_im]

Pole Class 2:   2 real poles,   0 cc pole pairs
[pos_re, coeff_re, coeff_re, 
 pos_re, coeff_re, coeff_re]

Pole Class 3:   1 real pole,    1 cc pole pair
[pos_re, coeff_re, coeff_re,
 pos_re, pos_im, coeff_re, coeff_im, coeff_re, coeff_im]

Pole Class 4:   0 real poles,   2 cc pole pairs
[pos_re, pos_im, coeff_re, coeff_im, coeff_re, coeff_im,
 pos_re, pos_im, coeff_re, coeff_im, coeff_re, coeff_im]

Pole Class 5:   3 real poles,   0 cc pole pairs
[pos_re, coeff_re, coeff_re,
 pos_re, coeff_re, coeff_re,
 pos_re, coeff_re, coeff_re]

Pole Class 6:   2 real poles,   1 cc pole pair
[pos_re, coeff_re, coeff_re,
 pos_re, coeff_re, coeff_re,
 pos_re, pos_im, coeff_re, coeff_im, coeff_re, coeff_im]

Pole Class 7:   1 real pole,    2 cc pole pairs
[pos_re, coeff_re, coeff_re,
 pos_re, pos_im, coeff_re, coeff_im, coeff_re, coeff_im,
 pos_re, pos_im, coeff_re, coeff_im, coeff_re, coeff_im]

Pole Class 8:   0 real poles,   3 cc pole pairs
[pos_re, pos_im, coeff_re, coeff_im, coeff_re, coeff_im,
 pos_re, pos_im, coeff_re, coeff_im, coeff_re, coeff_im,
 pos_re, pos_im, coeff_re, coeff_im, coeff_re, coeff_im]



