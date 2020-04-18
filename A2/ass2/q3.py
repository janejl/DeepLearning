
def learning_rate_range():
    """Give proper lower bound and upper bound for
    proper learning rate"""
    # Lower and upper bounds
    #######

    #######
    return lower_bound, upper_bound


def learnign_rate_examples():
    """Give three examples for a bad, not bad, and very good learning rate
    """
    #######

    #######
    return bad_larning_rate, not_bad_learning_rate, good_learning_rate


def test_learning_rate_range():
    lower, upper = learning_rate_range()
    assert isinstance(lower, float)
    assert isinstance(upper, float)
    print("\nPass.")


def test_learnign_rate_examples():
    bad_larning_rate, not_bad_learning_rate, good_learning_rate = learnign_rate_examples()
    assert isinstance(bad_larning_rate, float)
    assert isinstance(not_bad_learning_rate, float)
    assert isinstance(good_learning_rate, float)
    print("\nPass.")


if __name__ == '__main__':
    test_learning_rate_range()
    test_learnign_rate_examples()