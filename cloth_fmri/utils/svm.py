import matplotlib.pyplot as plt
import numpy as np
import random
from nilearn.decoding import Decoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample

from cloth_fmri.utils.stats import calc_p
from cloth_fmri.config.config import CONFIG


##### Model distance 
def get_woven_distance(verbose=False):
    from cloth_fmri.models.model_predictions import woven_stiff_pred
    random.seed(9)    
    woven_distance = {}
    for scene in CONFIG['scenes']:
        woven_soft = woven_stiff_pred[scene + '_0.5_0.0078125']
        woven_stiff = woven_stiff_pred[scene + '_0.5_2.0']
        random.shuffle(woven_soft)
        random.shuffle(woven_stiff)

        if len(woven_stiff) != len(woven_soft):
            min_length = min(len(woven_stiff), len(woven_soft))
            woven_stiff = woven_stiff[:min_length]
            woven_soft = woven_soft[:min_length]
        woven_distance[scene] = abs(np.array(woven_stiff) - np.array(woven_soft))
        
    if verbose:
        print("woven Distance:")
        for scene, distance in woven_distance.items():
            print(f"{scene}: {distance}")
        
    return woven_distance




def compute_soft_stiff_distance(pred_dict, scenes=None, verbose=False):
    """
    Compute absolute soft-vs-stiff prediction distance for each scene.

    Expected keys:
        "{scene}_0.5_0.0078125"  # soft
        "{scene}_0.5_2.0"        # stiff
    """
    if scenes is None:
        scenes = CONFIG["scenes"]

    distance = {}

    for scene in scenes:
        soft_key = f"{scene}_0.5_0.0078125"
        stiff_key = f"{scene}_0.5_2.0"

        soft = np.asarray(pred_dict[soft_key])
        stiff = np.asarray(pred_dict[stiff_key])

        if len(soft) != len(stiff):
            n = min(len(soft), len(stiff))
            soft = soft[:n]
            stiff = stiff[:n]

        distance[scene] = np.abs(stiff - soft)

    if verbose:
        for scene, values in distance.items():
            print(f"{scene}: {values}")

    return distance




def get_cnn_distance(verbose=False):
    from cloth_fmri.models.model_predictions import cnn_stiff_pred

    return compute_soft_stiff_distance(
        cnn_stiff_pred,
        verbose=verbose,
    )


def get_videomae_distance(verbose=False):
    from cloth_fmri.models.model_predictions import videomae_stiff_pred

    return compute_soft_stiff_distance(
        videomae_stiff_pred,
        verbose=verbose,
    )


def get_vivit_distance(verbose=False):
    from cloth_fmri.models.model_predictions import vivit_stiff_pred

    return compute_soft_stiff_distance(
        vivit_stiff_pred,
        verbose=verbose,
    )


def get_vjepa2_distance(verbose=False):
    from cloth_fmri.models.model_predictions import vjepa2_stiff_pred

    return compute_soft_stiff_distance(
        vjepa2_stiff_pred,
        verbose=verbose,
    )






class ManualSplit:
    def __init__(self, train_indices, test_indices):
        self.train_indices = train_indices
        self.test_indices = test_indices

    def split(self, X, y=None, groups=None):
        yield self.train_indices, self.test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1


def get_decoder(roi, manual_cv, smoothing_fwhm=0, C=1):
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
        cv=manual_cv,
    )

    return decoder


def dict_to_xy(data):
    X_vals = []
    y = []

    for key, values in data.items():
        if "2.0" in key:
            label = 1
        elif "0.0078125" in key:
            label = 0
        else:
            raise ValueError(
                f"{key} does not contain stiffness label"
            )

        for value in values:
            X_vals.append([float(value)])
            y.append(label)

    X = np.array(X_vals, dtype=float)
    y = np.array(y, dtype=int)

    return X, y


def _train_svm_core(
    X,
    y,
    test_size=0.5,
    random_state=42,
    n_boot=10000,
    plot=True,
    save_fig=False,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel="rbf",
                probability=False,
                random_state=random_state,
            ),
        ),
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {acc:.4f}")

    boot_accs = []
    n_test = len(X_test)

    for _ in range(n_boot):
        indices = np.random.choice(
            n_test,
            n_test,
            replace=True,
        )

        X_boot = X_test[indices]
        y_boot = y_test[indices]
        y_boot_pred = clf.predict(X_boot)

        boot_accs.append(
            accuracy_score(y_boot, y_boot_pred)
        )

    boot_accs = np.array(boot_accs)
    ci_lower = np.percentile(boot_accs, 2.5)
    ci_upper = np.percentile(boot_accs, 97.5)
    p = calc_p(boot_accs)

    print(
        f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}], "
        f"mean={np.mean(boot_accs):.3f}, p={p:.3f}"
    )

    if plot:
        plt.figure(figsize=(4, 6))
        plt.bar(
            0,
            acc,
            yerr=[[acc - ci_lower], [ci_upper - acc]],
            capsize=6,
            color="skyblue",
        )
        plt.xticks([0], ["Overall"])
        plt.ylabel("Accuracy")
        plt.title("Classifier Accuracy with 95% CI (bootstrap)")
        plt.ylim(0, 1)
        plt.tight_layout()

        if save_fig:
            output_file = "svm_accuracy.pdf"
            plt.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight",
            )
            print(f"Saved: {output_file}")

        plt.show()

    return {
        "model": clf,
        "accuracy": acc,
        "ci": (ci_lower, ci_upper),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def train_svm_classifier(
    data_or_X,
    y=None,
    test_size=0.5,
    random_state=42,
    n_boot=10000,
    plot=True,
    save_fig=False,
):
    if y is None:
        X, y = dict_to_xy(data_or_X)
    else:
        X = data_or_X

    return _train_svm_core(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        n_boot=n_boot,
        plot=plot,
        save_fig=save_fig,
    )


def train_svm_classifier_xy(
    X_train,
    X_test,
    y_train,
    y_test,
    random_state=42,
    n_boot=10000,
    plot=True,
    save_fig=False,
):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel="rbf",
                probability=False,
                random_state=random_state,
            ),
        ),
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {acc:.4f}")

    boot_accs = []

    for _ in range(n_boot):
        X_boot, y_boot = resample(
            X_test,
            y_test,
            replace=True,
        )
        y_boot_pred = clf.predict(X_boot)
        boot_accs.append(
            accuracy_score(y_boot, y_boot_pred)
        )

    boot_accs = np.array(boot_accs)
    ci_lower = np.percentile(boot_accs, 2.5)
    ci_upper = np.percentile(boot_accs, 97.5)
    p = calc_p(boot_accs)

    print(
        f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}], "
        f"mean={np.mean(boot_accs):.3f}, p={p:.3f}"
    )

    if plot:
        plt.figure(figsize=(4, 6))
        plt.bar(
            0,
            acc,
            yerr=[[acc - ci_lower], [ci_upper - acc]],
            capsize=6,
            color="skyblue",
        )
        plt.xticks([0], ["Overall"])
        plt.ylabel("Accuracy")
        plt.title("Classifier Accuracy with 95% CI (bootstrap)")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

    return {
        "model": clf,
        "accuracy": acc,
        "ci": (ci_lower, ci_upper),
        "y_pred": y_pred,
    }
