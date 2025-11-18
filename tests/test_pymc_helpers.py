import pymc as pm

from src.common.pymc_helpers import sample_nuts


def test_sample_nuts_smoke():
    """Ensure the shared wrapper can draw a few samples without raising."""
    with pm.Model():
        mu = pm.Normal("mu", 0.0, 1.0)
        pm.Normal("obs", mu=mu, sigma=1.0, observed=0.0)
        idata = sample_nuts(
            draws=5,
            tune=5,
            chains=1,
            cores=1,
            target_accept=0.8,
            progressbar=False,
            random_seed=123,
        )
    # Posterior should contain the latent variable and at least one sample.
    assert "mu" in idata.posterior
    assert idata.posterior["mu"].size > 0
