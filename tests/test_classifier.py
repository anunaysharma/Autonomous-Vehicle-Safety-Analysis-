from avsa import classifier, data


def test_assign_class_only_keeps_known_causes():
    df_diseng, _ = data.load_all()
    df = classifier.assign_class(df_diseng)
    assert (
        df["class"].isin(["Controller", "Perception System", "Computer System"]).all()
    )
    assert len(df) <= len(df_diseng)


def test_encode_features_shapes_match():
    df_diseng, _ = data.load_all()
    df = classifier.assign_class(df_diseng)
    X, y, encoders = classifier.encode_features(df)
    assert len(X) == len(y) == len(df)
    assert set(X.columns) == set(classifier.FEATURE_COLUMNS)


def test_train_and_evaluate_returns_valid_accuracy():
    df_diseng, _ = data.load_all()
    df = classifier.assign_class(df_diseng)
    X, y, _ = classifier.encode_features(df)
    result = classifier.train_and_evaluate(X, y)
    assert 0 <= result.accuracy <= 1
