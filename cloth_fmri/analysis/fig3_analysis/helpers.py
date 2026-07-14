from nilearn.decoding import Decoder
from sklearn.model_selection import LeaveOneGroupOut


def get_decoder(
    roi,
    leave_runs,
    manual_cv=None,
    smoothing_fwhm=5,
    C=1,
):

    if leave_runs == 1:
        cv = LeaveOneGroupOut()

    elif leave_runs == 2:
        cv = manual_cv

    else:
        raise ValueError("leave_runs must be either 1 or 2.")

    param_grid = [
        {
            "penalty": ["l1"],
            "dual": [False],
            "C": [float(C)],
        }
    ]

    decoder = Decoder(
        estimator="svc",
        smoothing_fwhm=smoothing_fwhm,
        standardize=True,
        mask=roi,
        screening_percentile=100,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
    )

    return decoder