# %%

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

%load_ext autoreload
%autoreload 2
# %%

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from numpyro import distributions

# %%

# %%

def sample_mixture(
    rng: jax.random.PRNGKeyArray, 
    means: jnp.ndarray, 
    var_isotropic_cov: float = 0.05,
    dim: int = 2,
    num_samples: int = 2 ** 14,
):
    covs = var_isotropic_cov * jnp.array(
        [
            jnp.eye(dim), 
            jnp.eye(dim)
        ]
        
    )
    mixture = distributions.MixtureSameFamily(
        mixing_distribution=distributions.Categorical(
            probs=jnp.ones(len(means)) / len(means)
        ),
        component_distribution=distributions.MultivariateNormal(
            loc=means, covariance_matrix=covs
        ),
    ) 
    samples, labels= mixture.sample_with_intermediates(
        rng, sample_shape=(num_samples,)
    )
    return samples, labels[0]
    
    
# %%

# source samples
source_means = jnp.array(
    [
        [-2, -2],
        [-2, 2]
    ]
)
source_samples, source_labels = sample_mixture(
    rng=jax.random.PRNGKey(0),
    means=source_means,
)
source_colors = np.array(
    [
        'blue' if lab == 0 else 'red' for lab in source_labels
    ]
)

# target samples
target_means = jnp.array(
    [
        [2, 2],
        [2, -2],
    ]
)
target_samples, target_labels = sample_mixture(
    rng=jax.random.PRNGKey(0),
    means=target_means,
)
target_colors = np.array(
    [
        'blue' if lab == 0 else 'red' for lab in target_labels
    ]
)

# %%

print(
    "Thanks to the choice of the PRNGkey, the samples are already aligned.",
    f"\nEquality of labels: {jnp.array_equal(source_labels, target_labels)}"
)


# %%

num_subsample = 256
indices_subsample = jax.random.choice(
    key=jax.random.PRNGKey(2), 
    a=jnp.arange(len(source_samples)), 
    replace=False,
    shape=(num_subsample,)
)

fig, ax = plt.subplots(figsize=(10,8))
plt.scatter(
    source_samples[indices_subsample, 0], 
    source_samples[indices_subsample, 1], 
    c=source_colors[indices_subsample], 
    edgecolors='k', s=200, marker='o'
)
plt.scatter(
    target_samples[indices_subsample, 0], 
    target_samples[indices_subsample, 1], 
    c=target_colors[indices_subsample],
    edgecolors='k', s=200, marker='X'
)
plt.title(
    "Circles = source; Crosses = target; Color = coupling.",
    fontsize=20
)
plt.show()

# %%

dataset = {
    "source": source_samples, 
    "target": target_samples,
    "labels": source_labels, # = target_labels since samples are aligned according to index
}
# %%
