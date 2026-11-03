"""Analytic checks for the renewal comparator at a two-second event horizon."""
import numpy as np
from audit_multihorizon import survival_risks

mixture = {'times': [1., 2.], 'survival': [.4, 0.]}
risk = survival_risks(np.array([.9, .5, 1.8]), mixture, 2, True)
np.testing.assert_allclose(risk, [.64, .4, .4], atol=1e-12)
np.testing.assert_allclose(
    survival_risks(np.array([.5, .9]), {'times': [1.], 'survival': [0.]}, 2, True),
    0, atol=1e-12)
np.testing.assert_allclose(
    survival_risks(np.array([.5]), {'times': [2.], 'survival': [0.]}, 2, True),
    1, atol=1e-12)
print('Analytic renewal checks passed.')
