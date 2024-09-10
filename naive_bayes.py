import numpy as np 

class GaussianNaiveBayes:
  """
  Gaussian Naive Bayes classifier for continuous data.

  Attributes:
      labels (list): List of unique class labels.
      means (dict): Dictionary with class labels as keys and mean of each feature in the class as values.
      variances (dict): Dictionary with class labels as keys and variance of each feature in the class as values.

  Methods:
      fit(X, y):
          Learn the mean and variance of each feature for each class.
      predict(X):
          Predict the class labels for the given data.

  Formulas:
      - Class Prior Probability:
        \[
        P(C) = \frac{N_C}{N}
        \]
        where \(N_C\) is the number of samples in class \(C\) and \(N\) is the total number of samples.

      - Feature Likelihoods (Normal Distribution):
        \[
        P(x_j \mid C) = \frac{1}{\sqrt{2 \pi \sigma_j^2}} \exp \left( -\frac{(x_j - \mu_j)^2}{2 \sigma_j^2} \right)
        \]
        where \( \mu_j \) and \( \sigma_j^2 \) are the mean and variance of feature \(x_j\) in class \(C\).

      - Posterior Probability:
        \[
        P(C \mid x) = \frac{P(C) \prod_{j=1}^{d} P(x_j \mid C)}{P(x)}
        \]
        where \(d\) is the number of features, and \(P(x)\) is the marginal likelihood.
  """
  
  def fit(self, X, y):
      self.n_features = X.shape[1]
      self.classes = np.unique(y)
      self.n_classes = len(self.classes)
  
      self.mean = np.zeros((self.n_classes, self.n_features))
      self.var = np.zeros((self.n_classes, self.n_features))
      self.priors = np.zeros(self.n_classes)
      
      for i, c in enumerate(self.classes):
          X_c = X[y == c]
          self.mean[i, :] = X_c.mean(axis=0)
          self.var[i, :] = X_c.var(axis=0)
          self.priors[i] = X_c.shape[0] / X.shape[0]
  
  def _gaussian_density(self, class_idx, x):
      mean = self.mean[class_idx]
      var = self.var[class_idx]
      numerator = np.exp(- (x - mean) ** 2 / (2 * var))
      denominator = np.sqrt(2 * np.pi * var)
      return numerator / denominator
  
  def predict(self, X):
      n_samples = X.shape[0]
      posteriors = np.zeros((n_samples, self.n_classes))
      
      for i, c in enumerate(self.classes):
          prior = np.log(self.priors[i])
          likelihood = np.sum(np.log(self._gaussian_density(i, X)), axis=1)
          posteriors[:, i] = prior + likelihood
      
      return self.classes[np.argmax(posteriors, axis=1)]


class MultinomialNaiveBayes:
  """
  Multinomial Naive Bayes classifier for discrete data.

  Attributes:
      labels (list): List of unique class labels.
      counts (dict): Dictionary with class labels as keys and count of each feature in the class as values.
      priors (dict): Dictionary with class labels as keys and prior probabilities of each class.

  Methods:
      fit(X, y, alpha=1.0):
          Learn the feature counts and class priors with optional smoothing.
      predict(X):
          Predict the class labels for the given data.

  Formulas:
      - Class Prior Probability:
        \[
        P(C) = \frac{N_C}{N}
        \]
        where \(N_C\) is the number of samples in class \(C\) and \(N\) is the total number of samples.

      - Feature Likelihoods (Multinomial Distribution):
        \[
        P(x_j \mid C) = \frac{N_{Cj} + \alpha}{N_C + \alpha \cdot |V|}
        \]
        where \(N_{Cj}\) is the count of feature \(j\) in class \(C\), \( \alpha \) is the smoothing parameter, and \(|V|\) is the vocabulary size.

      - Posterior Probability:
        \[
        P(C \mid x) = \frac{P(C) \prod_{j=1}^{d} P(x_j \mid C)}{P(x)}
        \]
        where \(d\) is the number of features, and \(P(x)\) is the marginal likelihood.
  """
  def fit(self, X, y, alpha=1.0):
      self.classes = np.unique(y)
      self.n_samples, self.n_features = X.shape
      self.feature_probs = []
      self.class_priors = []
      
      for c in self.classes:
          X_c = X[y == c]
          class_prior = X_c.shape[0] / self.n_samples
          class_feature_prob = (X_c.sum(axis=0) + alpha) / (X_c.sum() + alpha * self.n_features)
          self.feature_probs.append(class_feature_prob)
          self.class_priors.append(class_prior)
      
      self.feature_probs = np.array(self.feature_probs)
      self.class_priors = np.array(self.class_priors)
  
  def predict(self, X):
      posteriors = []
      
      for x in X:
          log_posteriors = []
          for idx, c in enumerate(self.classes):
              prior = np.log(self.class_priors[idx])
              likelihood = np.sum(np.log(self.feature_probs[idx]) * x)
              log_posteriors.append(prior + likelihood)
          posteriors.append(self.classes[np.argmax(log_posteriors)])
      
      return posteriors
      
class BernoulliNaiveBayes:
  """
    Bernoulli Naive Bayes classifier for binary/boolean data.
  
    Attributes:
        labels (list): List of unique class labels.
        feature_probs (dict): Dictionary with class labels as keys and probability of each feature being 1 in the class as values.
        class_priors (dict): Dictionary with class labels as keys and prior probabilities of each class.
  
    Methods:
        fit(X, y, alpha=1.0):
            Learn the feature probabilities and class priors with optional smoothing.
        predict(X):
            Predict the class labels for the given data.
  
    Formulas:
        - Class Prior Probability:
          \[
          P(C) = \frac{N_C}{N}
          \]
          where \(N_C\) is the number of samples in class \(C\) and \(N\) is the total number of samples.
  
        - Feature Likelihoods (Bernoulli Distribution):
          \[
          P(x_j \mid C) = \frac{N_{Cj} + \alpha}{N_C + \alpha \cdot 2}
          \]
          where \(N_{Cj}\) is the number of occurrences of feature \(j\) in class \(C\), and \(\alpha\) is the smoothing parameter.
  
        - Posterior Probability:
          \[
          P(C \mid x) = \frac{P(C) \prod_{j=1}^{d} P(x_j \mid C)}{P(x)}
          \]
          where \(d\) is the number of features, and \(P(x)\) is the marginal likelihood.
  """
  def fit(self, X, y, alpha=1.0):
      self.classes = np.unique(y)
      self.n_samples, self.n_features = X.shape
      self.feature_probs = []
      self.class_priors = []
      
      for c in self.classes:
          X_c = X[y == c]
          class_prior = X_c.shape[0] / self.n_samples
          feature_prob = (X_c.sum(axis=0) + alpha) / (X_c.shape[0] + alpha * 2)
          self.feature_probs.append(feature_prob)
          self.class_priors.append(class_prior)
      
      self.feature_probs = np.array(self.feature_probs)
      self.class_priors = np.array(self.class_priors)
  
  def predict(self, X):
      posteriors = []
      
      for x in X:
          log_posteriors = []
          for idx, c in enumerate(self.classes):
              prior = np.log(self.class_priors[idx])
              likelihood = np.sum(x * np.log(self.feature_probs[idx]) + (1 - x) * np.log(1 - self.feature_probs[idx]))
              log_posteriors.append(prior + likelihood)
          posteriors.append(self.classes[np.argmax(log_posteriors)])
      
      return posteriors


class ComplementNaiveBayes:
  """
  Complement Naive Bayes classifier for handling imbalanced datasets.

    Attributes:
        labels (list): List of unique class labels.
        feature_complement_probs (dict): Dictionary with class labels as keys and probability of each feature not being in the class as values.
        class_priors (dict): Dictionary with class labels as keys and prior probabilities of each class.

    Methods:
        fit(X, y, alpha=1.0):
            Learn the feature complement probabilities and class priors with optional smoothing.
        predict(X):
            Predict the class labels for the given data.

    Formulas:
        - Class Prior Probability:
          \[
          P(C) = \frac{N_C}{N}
          \]
          where \(N_C\) is the number of samples in class \(C\) and \(N\) is the total number of samples.

        - Feature Likelihoods (Complement Distribution):
          \[
          P(x_j \mid C) = \frac{N_{C'j} + \alpha}{N_{C'} + \alpha \cdot |V|}
          \]
          where \(N_{C'j}\) is the count of feature \(j\) in all classes except \(C\), \( \alpha \) is the smoothing parameter, and \(|V|\) is the vocabulary size.

        - Posterior Probability:
          \[
          P(C \mid x) = \frac{P(C) \prod_{j=1}^{d} P(x_j \mid C)}{P(x)}
          \]
          where \(d\) is the number of features, and \(P(x)\) is the marginal likelihood.
  """
  def fit(self, X, y, alpha=1.0):
      self.classes = np.unique(y)
      self.n_samples, self.n_features = X.shape
      self.class_priors = []
      self.feature_probs = []
      
      for c in self.classes:
          X_c = X[y != c]
          class_prior = X_c.shape[0] /  self.n_samples
          feature_complement_prob = (X_c.sum(axis=0) + alpha) / (X_c.sum() + alpha * self.n_features)
          self.feature_probs.append(feature_complement_prob)
          self.class_priors.append(class_prior)
      
      self.feature_probs = np.array(self.feature_probs)
      self.class_priors = np.array(self.class_priors)
  
  def predict(self, X):
      posteriors = []
      
      for x in X:
          log_posteriors = []
          for idx, c in enumerate(self.classes):
              prior = np.log(self.class_priors[idx])
              likelihood = np.sum(np.log(self.feature_probs[idx]) * x)
              log_posteriors.append(prior + likelihood)
          posteriors.append(self.classes[np.argmax(log_posteriors)])
      
      return posteriors
