# Information Constraints on Auto-Encoding Variational Bayes

Romain Lopez, Jeffrey Regier, Nir Yosef, Michael I. Jordan

University of California, Berkeley, EECS



Parameterizing the approximate posterior of a generative model with neural networks has become a common theme in recent machine learning research. While providing appealing flexibility, this approach makes it difficult to impose or assess structural constraints such as conditional independence. We propose a framework for learning representations that relies on Auto-Encoding Variational Bayes and whose search space is constrained via kernel-based measures of independence. In particular, our method employs the d-variable Hilbert-Schmidt Independence Criterion (dHSIC) to enforce independence between the latent representations and arbitrary nuisance factors. We show how to apply this method to a range of problems, including the problems of learning invariant representations and the learning of interpretable representations. We also present a full-fledged application to single-cell RNA sequencing (scRNA-seq). In this setting the biological signal in mixed in complex ways with sequencing errors and sampling effects. We show that our method out-performs the state-of-the-art in this domain.
