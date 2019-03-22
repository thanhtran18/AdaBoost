import numpy as np
from scipy.io import *
import matplotlib.pyplot as plt

def findWeakLearner(X,t,weights):
    # Fine the best weaker learner given data, targets, and weights.
    # weakl learner is a decision stump, p x(d) > p theta
    #
    # X is n-by-m-by-N
    # t is N-by-1
    # weight is N-by-1
    # d is 1-by-2 (index into 2-D matrix)
    # p is +/- 1
    # theta is scale
    # correct is N-by-1, binary. correct[i]=1 iff this weak learner correctly classifies example X[:,:,i]

    N=X.shape[2]
    
    t = np.squeeze(t)
    weights = np.squeeze(weights)
    
    # Sort all coordinates of X
    sinds = np.argsort(X) # sort based on last column
    Xs = np.sort(X)

    # Sort target vales according to this data sort
    Ts = t[sinds]

    # Sort weight values according to this data sort
    Ws = weights[sinds]

    # Compute cumsum to evalute goodness of possible thresholds theta.
    # cweight_pos[i,j,k] is amount of correct - incorrect weight incurred on left
    # side of threshold at (Xs[i,j,k]+Xs[i,j,k+1])/2
    cweight_pos = np.cumsum(Ts*Ws,2)

    # Do same in reverse (total -)
    # cwiehgt_neg[i,j,k] is amount of correct - incorrect weight incurred on right
    # side of threshold at (Xs[i,j,k]+Xs[i,j,k+1])/2
    cweight_neg = np.expand_dims(cweight_pos[:,:,-1],2) - cweight_pos

    # Max of either +1/1 times sum of two.
    signed_cor = cweight_pos - cweight_neg
    # Locations where Xs[i,j,k]==Xs[i,j,k+1] are not valid as possible threshold locations
    # Set these to zero so that we do not find thme as maxima
    valid = (np.diff(Xs,1,2) != 0)
    signed_cor = signed_cor * np.concatenate((valid,np.zeros((X.shape[0],X.shape[1],1))),2)

    us_cor = np.abs(signed_cor)

    i,j,k=np.unravel_index(np.argmax(us_cor), signed_cor.shape)

    # Compute theta, check boundary cayse
    if k==N-1:
        theta = np.inf
    else:
        theta = (Xs[i,j,k]+Xs[i,j,k+1])/2

    # The feature is i,j
    d=[i,j]
    
    # Check whether it was a +max or |-max| to get partity p
    p = -np.sign(signed_cor[i,j,k])

    # Whether or not this weak learner classifies examples correctly
    tmp=p*X[i,j,:]>p*theta
    correct = (t*(tmp-0.5))>0

    return d,p,theta,correct

def evaluateClassifier(classifier,x,t):
    # Evaluate classifer on data
    #
    # classifier is a map of alpha,d,p,theta arrays of length M
    # x is n-by-m-by-N, data
    # t is N-by-1, ground truth
    # errs is vector of length M
    #
    # errs[i] is error of classifier using first i components of committee in classifier

    M = len(classifier['alpha'])
    N = x.shape[2]

    # Responses f inputs to each classifier in committee
    resps = np.zeros((N,M))
    for m in range(M):
        x_use = x[classifier['d'][m,0],classifier['d'][m,1],:]
        resps[:,m]=classifier['alpha'][m]*np.sign((classifier['p'][m]*x_use > classifier['p'][m]*classifier['theta'][m])-0.5)

    # Compute output of classifier using first i components using cumsum
    class_out = np.sign(np.cumsum(resps,1))

    # Compare classifier output to ground-truth t
    correct = (class_out*np.reshape(t,(N,1)))>0

    errs = 1-np.mean(correct,0)

    return errs

def visualizeClassifier(classifier,fig_num,im_size):
    # Visualize a classifier
    # Color negative and positive pixels by sum of alpha values

    pos_feat = np.zeros(im_size)
    neg_feat = np.zeros(im_size)
    for m in range(len(classifier['alpha'])):
        if classifier['p'][m]>0:
            pos_feat[classifier['d'][m,0],classifier['d'][m,1]] += classifier['alpha'][m]
        else:
            neg_feat[classifier['d'][m,0],classifier['d'][m,1]] += classifier['alpha'][m]

    plt.figure(fig_num)
    plt.subplot(121)
    plt.imshow(pos_feat,cmap='gray')
    plt.title('Sum of weights on p=1 features')

    plt.subplot(122)
    plt.imshow(neg_feat,cmap='gray')
    plt.title('Sum of weights on p=-1 features')
    plt.show()
    
if __name__ == "__main__":
    N=5
    X=np.random.random((10,20,N))
    t=np.random.random((N,1))
    weights=np.random.random((N,1))

    d,p,theta,correct=findWeakLearner(X,t,weights)
    print(d)
    print(p)
    print(theta)
    print(correct)
