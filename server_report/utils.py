import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve, auc

import astropy.units as u
from scipy.stats import norm


def plot_error_histograms(err_protons, err_gammas):
    """
    Plot histograms of reconstruction errors for protons and gammas. Print mean, median, and std.
    """
    stats = {
        "Protons": {
            "mean": np.mean(err_protons),
            "median": np.median(err_protons),
            "std": np.std(err_protons)
        },
        "Gammas": {
            "mean": np.mean(err_gammas),
            "median": np.median(err_gammas),
            "std": np.std(err_gammas)
        }
    }

    for name, s in stats.items():
        print(f"{name}:")
        print(f"  Mean   = {s['mean']:.6f}")
        print(f"  Median = {s['median']:.6f}")
        print(f"  Std    = {s['std']:.6f}\n")

    all_errors = np.concatenate([err_protons, err_gammas])


    plt.figure(figsize=(8, 5))
    min_err = min(err_protons.min(), err_gammas.min())
    max_err = max(err_protons.max(), err_gammas.max())

    plt.hist(err_protons, bins=50, alpha=0.5, label="Protons", range=(min_err, max_err), density=True)
    plt.hist(err_gammas, bins=50, alpha=0.5, label="Gammas", range=(min_err, max_err), density=True)

    plt.axvline(np.median(all_errors), color="black", linestyle="--", label="Overall Median", alpha=0.5)

    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Density")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.show()

def evaluate_threshold_classifier(err_protons, err_gammas, threshold):
    """
    Evaluate a threshold-based classifier using reconstruction errors.
    """

    print("Using threshold:", threshold)

    y_true = np.concatenate([
        np.zeros_like(err_protons),     # Protons = 0
        np.ones_like(err_gammas)        # Gammas = 1
    ])

    y_pred = np.concatenate([
        (err_protons > threshold).astype(int),
        (err_gammas > threshold).astype(int)
    ])

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Protons", "Gammas"])

    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (threshold = {:.4f})".format(threshold))
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Protons", "Gammas"]))

    y_scores = np.concatenate([err_protons, err_gammas])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray") 
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return y_true, y_pred, cm

def clean_image_improvement(peak, mask, adj_list, max_diff=5):
    """
    Clean image by expanding the mask to neighboring pixels with similar peak values.
    """
    new_mask = mask.copy().astype(np.uint8)
    queue = deque(np.where(new_mask == 1)[0])

    while queue:
        i = queue.popleft()
        for n in adj_list[i]:
            if new_mask[n] == 0 and abs(peak[i] - peak[n]) <= max_diff:
                new_mask[n] = 1
                queue.append(n)

    return new_mask


def plot_cost_curves(train_loss_history, val_loss_history, epochs):
    """
    Plot training and validation cost curves over epochs.
    """

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_loss_history, label="Train Cost")
    plt.plot(range(1, epochs + 1), val_loss_history, label="Validation Cost")

    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Cost")
    plt.legend()
    plt.show()

def plot_feature_importances(model1, model2, features=None, label1="RF Original", label2="RF from AE Image", title="Feature importances comparison"):
    """
    Plot feature importances for two Random Forest models side by side.
    """

    importances1 = model1.feature_importances_
    std1 = np.std([tree.feature_importances_ for tree in model1.estimators_], axis=0)

    importances2 = model2.feature_importances_
    std2 = np.std([tree.feature_importances_ for tree in model2.estimators_], axis=0)

    # Order features by model1
    indices = np.argsort(importances1)
    ordered_features = [features[i] for i in indices]

    x = np.arange(len(features))  # positions for features
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.barh(x - width/2, importances1[indices], width, xerr=std1[indices], label=label1)
    ax.barh(x + width/2, importances2[indices], width, xerr=std2[indices], label=label2, color='orange')

    ax.set_yticks(x)
    ax.set_yticklabels(ordered_features)
    ax.set_xlabel("Importance score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='x')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_pred1, y_pred2, label1="RF 1", label2="RF 2"):
    """
    Plot ROC curves for two sets of predictions.
    """

    fpr1, tpr1, _ = roc_curve(y_true, y_pred1)
    auc1 = auc(fpr1, tpr1)

    fpr2, tpr2, _ = roc_curve(y_true, y_pred2)
    auc2 = auc(fpr2, tpr2)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr1, tpr1, label=f"{label1} (AUC = {auc1:.3f})")
    plt.plot(fpr2, tpr2, label=f"{label2} (AUC = {auc2:.3f})", color="orange")
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid(True)
    plt.show()

def plot_gammaness_distribution(y_true, y_scores, threshold=0.5, bins=50, title="Gammaness Probability Distribution"):
    """
    Plot the distribution of predicted gammaness probabilities for protons and gammas.
    """
    plt.figure(figsize=(8, 6))

    plt.hist(y_scores[y_true == 0], bins=bins, alpha=0.5, label='Protons', color='red', density=True)
    plt.hist(y_scores[y_true == 1], bins=bins, alpha=0.5, label='Gammas', color='blue', density=True)

    plt.axvline(x=threshold, color='black', linestyle='--', label='Threshold')

    plt.xlabel('Predicted Gammaness probability')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_energy_resolution(y_true, y_pred1, y_pred2,
                           label1="RF 1", label2="RF 2",
                           n_bins=10):
    """
    Plot energy resolution curves for two sets of energy predictions.
    """

    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)

    y_true = 10 ** y_true
    y_pred1 = 10 ** y_pred1
    y_pred2 = 10 ** y_pred2

    # bins en log
    bins = np.logspace(np.log10(y_true.min()),
                       np.log10(y_true.max()),
                       n_bins + 1)

    centers = np.sqrt(bins[:-1] * bins[1:])

    def resolution(y_t, y_p):
        res = []
        for i in range(len(bins) - 1):
            mask = (y_t >= bins[i]) & (y_t < bins[i + 1])
            if np.sum(mask) < 5:
                res.append(np.nan)
                continue

            rel = (y_p[mask] - y_t[mask]) / y_t[mask]
            p16, p84 = np.percentile(rel, [16, 84])
            res.append(0.5 * (p84 - p16))
        return np.array(res)

    r1 = resolution(y_true, y_pred1)
    r2 = resolution(y_true, y_pred2)

    plt.figure(figsize=(6, 4))
    plt.plot(centers, r1, "o-", label=label1)
    plt.plot(centers, r2, "s-", label=label2)
    plt.ylim(0, 1)

    plt.xscale("log")
    plt.xlabel("E_true")
    plt.ylabel("(ΔE / E_true)$_{68}$")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

def transform_pred_to_labels(y_proba, threshold=0.5):
    """
    Transform predicted probabilities to binary labels based on a threshold.
    """
    return (y_proba >= threshold).astype(int)

# Source: https://github.com/cta-observatory/ctaplot/blob/master/ctaplot/ana/ana.py#L1075
def bias(true, reco):
    """
    Compute the bias of a reconstructed variable as `median(reco-true)`

    Parameters
    ----------
    true: `numpy.ndarray`
    reco: `numpy.ndarray`

    Returns
    -------
    float
    """
    if len(true) != len(reco):
        raise ValueError("both arrays should have the same size")
    if len(true) == 0:
        return 0
    return np.median(reco - true)

# Source: https://github.com/cta-observatory/ctaplot/blob/master/ctaplot/ana/ana.py#L1075
def relative_scaling(true, reco, method='s0'):
    """
    Define the relative scaling for the relative error calculation.
    There are different ways to calculate this scaling factor.
    The easiest and most spread one is simply `np.abs(true)`. However this is possible only when `true != 0`.
    Possible methods:
    - None or 's0': scale = 1
    - 's1': `scale = np.abs(true)`
    - 's2': `scale = np.abs(reco)`
    - 's3': `scale = (np.abs(true) + np.abs(reco))/2.`
    - 's4': `scale = np.max([np.abs(reco), np.abs(true)], axis=0)`

    This method is not exposed but kept for tests and future reference.
    The `s1` method is used in all `ctaplot` functions.

    Parameters
    ----------
    true: `numpy.ndarray`
    reco: `numpy.ndarray`

    Returns
    -------
    `numpy.ndarray`
    """
    method = 's0' if method is None else method
    scaling_methods = {
        's0': lambda true, reco: np.ones(len(true)),
        's1': lambda true, reco: np.abs(true),
        's2': lambda true, reco: np.abs(reco),
        's3': lambda true, reco: (np.abs(true) + np.abs(reco)) / 2.,
        's4': lambda true, reco: np.max([np.abs(reco), np.abs(true)], axis=0)
    }

    return scaling_methods[method](true, reco)

# Source: https://github.com/cta-observatory/ctaplot/blob/master/ctaplot/ana/ana.py#L1075
def _percentile(x, percentile=68.27):
    """
    Compute the value of the Qth containment radius
    Return 0 if the list is empty
    Parameters
    ----------
    x: numpy array or list

    Returns
    -------
    float
    """
    if len(x) != 0:
        return np.percentile(x, percentile)
    if isinstance(x, u.Quantity):
        return 0 * x.unit
    else:
        return 0
    
# Source: https://github.com/cta-observatory/ctaplot/blob/master/ctaplot/ana/ana.py#L1075
def percentile_confidence_interval(x, percentile=68, confidence_level=0.95):
    """
    Return the confidence interval for the qth percentile of x for a given confidence level

    REF:
    http://people.stat.sfu.ca/~cschwarz/Stat-650/Notes/PDF/ChapterPercentiles.pdf
    S. Chakraborti and J. Li, Confidence Interval Estimation of a Normal Percentile, doi:10.1198/000313007X244457

    Parameters
    ----------
    x: `numpy.ndarray`
    percentile: `float`
        0 < percentile < 100
    confidence_level: `float`
        0 < confidence level (by default 95%) < 1

    Returns
    -------

    """
    sorted_x = np.sort(x)
    z = norm.ppf(confidence_level)
    if len(x) == 0:
        return 0, 0
    q = percentile / 100.

    j = np.max([0, int(len(x) * q - z * np.sqrt(len(x) * q * (1 - q)))])
    k = np.min([int(len(x) * q + z * np.sqrt(len(x) * q * (1 - q))), len(x) - 1])
    return sorted_x[j], sorted_x[k]

# Source: https://github.com/cta-observatory/ctaplot/blob/master/ctaplot/ana/ana.py#L1075
def resolution(true, reco,
               percentile=68.27, confidence_level=0.95, bias_correction=False, relative_scaling_method='s1'):
    """
    Compute the resolution of reco as the Qth (68.27 as standard = 1 sigma) containment radius of
    `(true-reco)/relative_scaling` with the lower and upper confidence limits defined the values inside
    the error_percentile

    Parameters
    ----------
    true: `numpy.ndarray` (1d)
        simulated quantity
    reco: `numpy.ndarray` (1d)
        reconstructed quantity
    percentile: float
        percentile for the resolution containment radius
    error_percentile: float
        percentile for the confidence limits
    bias_correction: bool
        if True, the resolution is corrected with the bias computed on true and reco
    relative_scaling: str
        see `ctaplot.ana.relative_scaling`

    Returns
    -------
    `numpy.ndarray` - [resolution, lower_confidence_limit, upper_confidence_limit]
    """
    assert len(true) == len(reco), "both arrays should have the same size"

    b = bias(true, reco) if bias_correction else 0

    with np.errstate(divide='ignore', invalid='ignore'):
        reco_corr = reco - b
        res = np.nan_to_num(np.abs((reco_corr - true) /
                                   relative_scaling(true, reco_corr, method=relative_scaling_method)))

    return np.append(_percentile(res, percentile), percentile_confidence_interval(res, percentile=percentile,
                                                                                  confidence_level=confidence_level))


# Source: https://github.com/cta-observatory/ctaplot/blob/master/ctaplot/ana/ana.py#L1075
def resolution_per_bin(x, y_true, y_reco,
                       percentile=68.27,
                       confidence_level=0.95,
                       bias_correction=False,
                       relative_scaling_method=None,
                       bins=10):
    """
    Resolution of y as a function of binned x.

    Parameters
    ----------
    x: `numpy.ndarray`
    y_true: `numpy.ndarray`
    y_reco: `numpy.ndarray`
    percentile: float
    confidence_level: float
    bias_correction: bool
    relative_scaling_method: see `ctaplot.ana.relative_scaling`
    bins: int or `numpy.ndarray` (see `numpy.histogram`)

    Returns
    -------
    (x_bins, res): (`numpy.ndarray`, `numpy.ndarray`)
        x_bins: bins for x
        res: resolutions with confidence level intervals for each bin
    """
    _, x_bins = np.histogram(x, bins=bins)
    bin_index = np.digitize(x, x_bins)
    res = []
    for ii in np.arange(1, len(x_bins)):
        mask = bin_index == ii
        res.append(resolution(y_true[mask], y_reco[mask],
                              percentile=percentile,
                              confidence_level=confidence_level,
                              relative_scaling_method=relative_scaling_method,
                              bias_correction=bias_correction,
                              )
                   )

    res = u.Quantity(res) if isinstance(res[0], u.Quantity) else np.array(res)
    return x_bins, res

# Source: https://github.com/cta-observatory/ctaplot/blob/master/ctaplot/ana/ana.py#L1075
def relative_bias(true, reco, relative_scaling_method='s1'):
    """
    Compute the relative bias of a reconstructed variable as
    `median(reco-true)/relative_scaling(true, reco)`

    Parameters
    ----------
    true: `numpy.ndarray`
    reco: `numpy.ndarray`
    relative_scaling_method: str
        see `ctaplot.ana.relative_scaling`

    Returns
    -------

    """
    assert len(reco) == len(true)
    if len(true) == 0:
        return 0
    return np.median((reco - true) / relative_scaling(true, reco, method=relative_scaling_method))


# Source: https://github.com/cta-observatory/ctaplot/blob/master/ctaplot/ana/ana.py#L1075
def bias_per_bin(true, reco, x, relative_scaling_method=None, bins=10):
    """
    Bias between `true` and `reco` per bin of `x`.

    Parameters
    ----------
    true: `numpy.ndarray`
    reco: `numpy.ndarray`
    x: : `numpy.ndarray`
    relative_scaling_method: str
        see `ctaplot.ana.relative_scaling`
    bins: bins for `numpy.histogram`

    Returns
    -------
    bins, bias: `numpy.ndarray, numpy.ndarray`
    """
    _, x_bins = np.histogram(x, bins=bins)
    bin_index = np.digitize(x, x_bins)
    b = []
    for ii in np.arange(1, len(x_bins)):
        mask = bin_index == ii
        b.append(relative_bias(true[mask], reco[mask], relative_scaling_method=relative_scaling_method))

    b = u.Quantity(b) if isinstance(b[0], u.Quantity) else np.array(b)
    return x_bins, b

def plot_resolution_two_curves(
    x_bins_1, res1,
    x_bins_2, res2,
    label1="Curve 1",
    label2="Curve 2",
    xlabel="Energy",
    ylabel="Resolution",
    title="Energy Resolution Comparison"
):
    """
    Plot two resolution curves for comparison.
    """
    
    x_bins_1 = 10**x_bins_1
    x_bins_2 = 10**x_bins_2


    # Centres des bins
    x1 = 0.5 * (x_bins_1[:-1] + x_bins_1[1:])
    x2 = 0.5 * (x_bins_2[:-1] + x_bins_2[1:])

    y1 = res1[:, 0]
    y2 = res2[:, 0]

    plt.figure(figsize=(8, 6))

    plt.plot(x1, y1, "--o", label=label1)
    plt.plot(x2, y2, "--s", label=label2)

    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_xticks([1e0,1e1, 1e2])

    plt.ylim(0, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.title(title)

    plt.tight_layout()
    plt.show()
