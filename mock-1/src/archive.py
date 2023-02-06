
# Global minimization

class BoundedHop:
    def __init__(self, k_lower, k_upper, stepsize=10):
        self.k_lower = k_lower
        self.k_upper = k_upper
        self.stepsize = stepsize
    def __call__(self, k):
        min_step = np.maximum(self.k_lower - k, -self.stepsize)
        max_step = np.minimum(self.k_upper - k, self.stepsize)
        random_step = np.random.uniform(low=min_step, high=max_step, size=k.shape)
        # print(' k_current = ', k, '\n min_step = ', min_step, '\n max_step = ', max_step, '\n Random step = ', random_step)
        k += random_step
        # print(' k_new = ', k, '\n')
        return k


class AcceptBounds:
    def __init__(self, k_lower, k_upper):
        self.k_lower = k_lower
        self.k_upper = k_upper
    def __call__(self, **kwargs):
        k = kwargs["x_new"]
        # print('k (accept bounds) =', k)
        k_max_bool = bool(np.all(k <= self.k_upper))
        k_min_bool = bool(np.all(k >= self.k_lower))
        accept = k_max_bool and k_min_bool
        # print('Accept bounds =', accept, '\n')
        return accept