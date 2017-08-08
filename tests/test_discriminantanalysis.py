import ML.discriminantanalysis as discriminentanalysis
import data
import numpy as np

def test_LDA():
    X, y = data.categorical_2Dmatrix_data()
    lda = discriminentanalysis.DiscriminentAnalysis(alpha=0)
    lda.fit(X, y)
    prediction0 = lda.predict(X[0])
    prediction1 = lda.predict(X[-1])
    assert prediction0 == y[0]
    assert prediction1 == y[-1]


def test_QDA():
    X, y = data.categorical_2Dmatrix_data()
    qda = discriminentanalysis.DiscriminentAnalysis(alpha=1)
    qda.fit(X, y)
    prediction0 = qda.predict(X[0])
    prediction1 = qda.predict(X[-1])
    assert prediction0 == 1
    assert prediction1 == y[-1]


def test_RDA():
    X, y = data.categorical_2Dmatrix_data()
    rda = discriminentanalysis.DiscriminentAnalysis(alpha=0.5)
    rda.fit(X, y)
    prediction0 = rda.predict(X[1])
    prediction1 = rda.predict(X[-1])
    assert prediction0 == y[0]
    assert prediction1 == y[-1]
