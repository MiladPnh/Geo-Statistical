import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class GMM:
    """ Gaussian Mixture Model

    Parameters
    -----------
        k: int , number of gaussian distributions

        seed: int, will be randomly set if None

        max_iter: int, number of iterations to run algorithm, default: 200

    Attributes
    -----------
       centroids: array, k, number_features

       cluster_labels: label for each data point

    """

    def __init__(self, C, n_runs):
        self.C = C  # number of Guassians/clusters
        self.n_runs = n_runs

    def get_params(self):
        return (self.mu, self.pi, self.sigma)

    def calculate_mean_covariance(self, X, prediction):
        """Calculate means and covariance of different
            clusters from k-means prediction

        Parameters:
        ------------
        prediction: cluster labels from k-means

        X: N*d numpy array data points

        Returns:
        -------------
        intial_means: for E-step of EM algorithm

        intial_cov: for E-step of EM algorithm

        """
        d = X.shape[1]
        labels = np.unique(prediction)
        self.initial_means = np.zeros((self.C, d))
        self.initial_cov = np.zeros((self.C, d, d))
        self.initial_pi = np.zeros(self.C)

        counter = 0
        for label in labels:
            ids = np.where(prediction == label)  # returns indices
            self.initial_pi[counter] = len(ids[0]) / X.shape[0]
            self.initial_means[counter, :] = np.mean(X[ids], axis=0)
            de_meaned = X[ids] - self.initial_means[counter, :]
            Nk = X[ids].shape[0]  # number of data points in current gaussian
            self.initial_cov[counter, :, :] = np.dot(self.initial_pi[counter] * de_meaned.T, de_meaned) / Nk
            counter += 1
        assert np.sum(self.initial_pi) == 1

        return (self.initial_means, self.initial_cov, self.initial_pi)

    def _initialise_parameters(self, X):
        """Implement k-means to find starting
            parameter values.
            https://datascience.stackexchange.com/questions/11487/how-do-i-obtain-the-weight-and-variance-of-a-k-means-cluster
        Parameters:
        ------------
        X: numpy array of data points

        Returns:
        ----------
        tuple containing initial means and covariance

        _initial_means: numpy array: (C*d)

        _initial_cov: numpy array: (C,d*d)


        """
        n_clusters = self.C
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=500, algorithm='auto')
        fitted = kmeans.fit(X)
        prediction = kmeans.predict(X)
        self._initial_means, self._initial_cov, self._initial_pi = self.calculate_mean_covariance(X, prediction)

        return (self._initial_means, self._initial_cov, self._initial_pi)

    def _e_step(self, X, pi, mu, sigma):
        """Performs E-step on GMM model
        Parameters:
        ------------
        X: (N x d), data points, m: no of features
        pi: (C), weights of mixture components
        mu: (C x d), mixture component means
        sigma: (C x d x d), mixture component covariance matrices
        Returns:
        ----------
        gamma: (N x C), probabilities of clusters for objects
        """
        N = X.shape[0]
        self.gamma = np.zeros((N, self.C))

        const_c = np.zeros(self.C)

        self.mu = self.mu if self._initial_means is None else self._initial_means
        self.pi = self.pi if self._initial_pi is None else self._initial_pi
        self.sigma = self.sigma if self._initial_cov is None else self._initial_cov

        for c in range(self.C):
            # Posterior Distribution using Bayes Rule
            self.gamma[:, c] = self.pi[c] * mvn.pdf(X, self.mu[c, :], self.sigma[c])

        # normalize across columns to make a valid probability
        gamma_norm = np.sum(self.gamma, axis=1)[:, np.newaxis]
        self.gamma /= gamma_norm

        return self.gamma

    def _m_step(self, X, gamma):
        """Performs M-step of the GMM
        We need to update our priors, our means
        and our covariance matrix.
        Parameters:
        -----------
        X: (N x d), data
        gamma: (N x C), posterior distribution of lower bound
        Returns:
        ---------
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)
        """
        N = X.shape[0]  # number of objects
        C = self.gamma.shape[1]  # number of clusters
        d = X.shape[1]  # dimension of each object

        # responsibilities for each gaussian
        self.pi = np.mean(self.gamma, axis=0)

        self.mu = np.dot(self.gamma.T, X) / np.sum(self.gamma, axis=0)[:, np.newaxis]

        for c in range(C):
            x = X - self.mu[c, :]  # (N x d)

            gamma_diag = np.diag(self.gamma[:, c])
            x_mu = np.matrix(x)
            gamma_diag = np.matrix(gamma_diag)

            sigma_c = x.T * gamma_diag * x
            self.sigma[c, :, :] = (sigma_c) / np.sum(self.gamma, axis=0)[:, np.newaxis][c]

        return self.pi, self.mu, self.sigma

    def _compute_loss_function(self, X, pi, mu, sigma):
        """Computes lower bound loss function

        Parameters:
        -----------
        X: (N x d), data

        Returns:
        ---------
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)
        """
        N = X.shape[0]
        C = self.gamma.shape[1]
        self.loss = np.zeros((N, C))

        for c in range(C):
            dist = mvn(self.mu[c], self.sigma[c], allow_singular=True)
            self.loss[:, c] = self.gamma[:, c] * (
                        np.log(self.pi[c] + 0.00001) + dist.logpdf(X) - np.log(self.gamma[:, c] + 0.000001))
        self.loss = np.sum(self.loss)
        return self.loss

    def fit(self, X):
        """Compute the E-step and M-step and
            Calculates the lowerbound

        Parameters:
        -----------
        X: (N x d), data

        Returns:
        ----------
        instance of GMM

        """

        d = X.shape[1]
        self.mu, self.sigma, self.pi = self._initialise_parameters(X)

        try:
            for run in range(self.n_runs):
                self.gamma = self._e_step(X, self.mu, self.pi, self.sigma)
                self.pi, self.mu, self.sigma = self._m_step(X, self.gamma)
                loss = self._compute_loss_function(X, self.pi, self.mu, self.sigma)

                if run % 10 == 0:
                    print("Iteration: %d Loss: %0.6f" % (run, loss))


        except Exception as e:
            print(e)

        return self

    def predict(self, X):
        """Returns predicted labels using Bayes Rule to
        Calculate the posterior distribution

        Parameters:
        -------------
        X: ?*d numpy array

        Returns:
        ----------
        labels: predicted cluster based on
        highest responsibility gamma.

        """
        labels = np.zeros((X.shape[0], self.C))

        for c in range(self.C):
            labels[:, c] = self.pi[c] * mvn.pdf(X, self.mu[c, :], self.sigma[c])
        labels = labels.argmax(1)
        return labels

    def predict_proba(self, X):
        """Returns predicted labels

        Parameters:
        -------------
        X: N*d numpy array

        Returns:
        ----------
        labels: predicted cluster based on
        highest responsibility gamma.

        """
        post_proba = np.zeros((X.shape[0], self.C))

        for c in range(self.C):
            # Posterior Distribution using Bayes Rule, try and vectorise
            post_proba[:, c] = self.pi[c] * mvn.pdf(X, self.mu[c, :], self.sigma[c])

        return post_proba



def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))



df = pd.read_csv('11.csv').replace(' ', '')
df = df.apply(pd.to_numeric)
features_names = df.columns.values.tolist()
X = df.drop('Class', axis=1)
sklearn_pca = PCA(n_components = 2)
Y_sklearn = sklearn_pca.fit_transform(X)
type(Y_sklearn)



for i in range(22):
    result = df[[features_names[i], 'Area Growth Rate']]
    result = result.values
    model = GMM(3, n_runs=30)
    fitted_values = model.fit(result)
    predicted_values = model.predict(Y_sklearn)
    centers = np.zeros((3, 2))
    for i in range(model.C):
        density = mvn(cov=model.sigma[i], mean=model.mu[i]).logpdf(Y_sklearn)
        centers[i, :] = Y_sklearn[np.argmax(density)]

    plt.figure(figsize=(10, 10))
    plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=predicted_values, s=200, cmap='viridis', zorder=1)

    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=1000, alpha=0.5, zorder=2)

    w_factor = 0.2 / model.pi.max()
    for pos, covar, w in zip(model.mu, model.sigma, model.pi):
        draw_ellipse(pos, covar, alpha=w)

    print("My model weights: ", model.pi)
    print("My model means: ", model.mu)
    print("My model Covariance: ", model.sigma)
    plt.show()


#Main#

from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.decomposition import PCA

df = pd.read_csv('22.csv').replace(' ', '')
df = df.apply(pd.to_numeric)

#X = df.drop('Class', axis=1)
X = df.drop('Area Growth Rate', axis=1)
y = df['Area Growth Rate']
y = np.asarray(y)
y = (y-min(y))/max(y)
features_names = df.columns.values.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)


#After Feature Extraction
result = df[[features_names[4], features_names[3],features_names[20],features_names[5],features_names[18]]]
result = result.values
sklearn_pca = PCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(result)


from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=5, covariance_type='full').fit(Y_sklearn)
prediction_gmm = gmm.predict(Y_sklearn)
probs = gmm.predict_proba(Y_sklearn)


centers = np.zeros((5,2))
for i in range(5):
    density = mvn(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(result)
    centers[i, :] = result[np.argmax(density)]

plt.figure(figsize = (10,8))
# plt.xlabel(features_names[i])
# plt.ylabel(features_names[i+5])
plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],c=prediction_gmm ,s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6)
for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
    draw_ellipse(pos, covar, alpha=w)
plt.show()


n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(Y_sklearn)
          for n in n_components]
plt.figure(figsize=(20,20))
plt.plot(n_components, [m.bic(Y_sklearn) for m in models], label='BIC')
plt.plot(n_components, [m.aic(Y_sklearn) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');
plt.show()