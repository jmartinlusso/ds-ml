import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

data = load_iris()

candidates = [
    ("LogReg", lambda: LogisticRegression(max_iter=200)),
    ("Tree depth=1", lambda: DecisionTreeClassifier(max_depth=1, random_state=42)),
    ("Tree depth=2", lambda: DecisionTreeClassifier(max_depth=2, random_state=42)),
    ("Tree depth=3", lambda: DecisionTreeClassifier(max_depth=3, random_state=42)),
    ("Tree depth=None", lambda: DecisionTreeClassifier(random_state=42)),
]

for name, make_model in candidates:
    train_scores, test_scores = [], []
    for seed in range(20):
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=seed
        )
        model = make_model()
        model.fit(X_train, y_train)
        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))
    tr, te = np.mean(train_scores), np.mean(test_scores)
    print(f"{name:16} | train={tr:.3f} | test={te:.3f} | gap={tr - te:+.3f}")