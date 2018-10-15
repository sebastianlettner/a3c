import scipy.signal


def discount(x, gamma):
    """
    This function helps computing the discounted reward for a list of rewards.
    For a given list c = [1, 1, 1, 3] the output list elements are computed the following way:
        Starting from the last element x4 = 3, x3 is computed as x3 + gamma*x4
        Then x2 is computed as x2 + gamma*x3 and so on.

    Args:
        x(ndarray): List of elemnt you want to apply discount on
        gamma(float): Discount factor

    Returns:
        List with discount applied on the elements
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

