# Robust Hybrid Physics–ML Modeling for Forward Osmosis Water Flux Prediction

This repository provides a complete workflow for **modeling**, **predicting**, and **quantifying uncertainty** in forward-osmosis (FO) water flux using both purely data-driven and physics-informed machine-learning models. The repository implements:

1. **Pure ANN (MLP) Model**
2. **Hybrid ANN Model** (Physics + ANN residual correction)
3. **Robust GPR Model** (Gaussian Process Regression with input-uncertainty propagation)
4. **Hybrid Robust GPR Model** (Physics + GPR residual + full uncertainty propagation)

All pipelines produce accuracy metrics, uncertainty estimates, diagnostic analyses, and plots.

---

# 1. Overview

Forward osmosis (FO) water flux depends on multiple physical, geometric, and membrane parameters. Direct numerical modeling is computationally expensive, and measurements include uncertainty. This repository develops **surrogate models** that are:

- **Accurate** — hybrid physics–ML improves predictive performance.
- **Data-efficient** — residual learning reduces model complexity.
- **Uncertainty-aware** — input and model uncertainty are rigorously propagated.
- **Validated** — Monte-Carlo simulation confirms analytical uncertainty estimates.

The goal is to create reliable models for simulation, optimization, and scientific analysis.

---

# 2. Data Preprocessing and Units

The dataset (`results_final.csv`) includes FO operating conditions and measured water flux. All pipelines apply:

### • Automatic removal of index/Unnamed columns  
### • Conversion to SI units:
- `Lx (mm)` → m  
- `t_c (mm)` → m  
- `t_psl (um)` → m  
- `Water flux (um/s)` → m/s  

### • Feature set (10 inputs)
```
cf_in (M)
cd_in (M)
Lx (m)
uf_in (m/s)
ud_in (m/s)
t_c (m)
t_psl (m)
eps_psl
tau
A (m/Pa/s)
```

---

# 3. Physics Model for Forward Osmosis

A forward-osmosis physical model is implemented to compute the water flux `jw_physical` by solving a nonlinear equation using **Brent’s method**.

The model incorporates:
- Osmotic pressures  
- Diffusion coefficients  
- Concentration polarization  
- Mass-transfer coefficients (feed and draw)  
- Membrane structural parameter effects  
- Transport coefficient `A`

Given inputs \(X\), the physics solver returns:

\[
j_{w,\text{physical}}(X)
\]

This is used directly in hybrid models.

---

# 4. Machine-Learning Pipelines

## 4.1 Pure ANN Model (MLPRegressor)

A direct data-driven regression model trained on standardized inputs and standardized target.

**Steps:**
1. Train/test split (90% test set for fixed evaluation)  
2. StandardScaler for X and y  
3. MLP with architecture (50, 25)  
4. Early stopping enabled  
5. Predictions transformed back to physical units  

**Outputs:**  
- MAE, R², MAPE  
- Parity plot  
- Saved model + scalers (`joblib`)

---

## 4.2 Hybrid ANN Model (Physics + ANN Residual)

The ANN learns **only the residual** between the measured flux and the physics-based prediction:

\[
e = y_\text{actual} - j_{w,\text{physical}}
\]

Final prediction:

\[
j_{w,\text{hybrid}} = j_{w,\text{physical}} + e_\text{ANN}
\]

**Outputs:**  
- Physics-only vs hybrid comparison  
- Parity plots  
- Residual histograms  
- Saved model + scalers

---

# 5. Robust GPR with Input-Uncertainty Propagation

A Gaussian Process Regression (GPR) model is trained on the standardized inputs and outputs. The model also propagates **measurement uncertainty** in inputs to the output using the **Delta method**.

## 5.1 GPR Model
Kernel:
- Constant × RBF  
- Output normalized  
- Hyperparameters optimized with restarts  

GPR yields:
- Predictive mean  
- Predictive variance (“model uncertainty”)

## 5.2 Input Uncertainty Modeling

Each input feature has a known coefficient of variation (CV). A full input covariance matrix \(\Sigma_z\) is built in standardized (z) space:

\[
\Sigma_z = D_z \, \text{Corr} \, D_z
\]

where:
- \(D_z\) is feature uncertainty in z-space  
- `Corr` is a feature-correlation matrix  

## 5.3 Gradient-Based Delta Method

For each test sample, gradients are computed numerically:

\[
g_i = \frac{\partial f}{\partial z_i}
\]

Input-propagated output variance:

\[
\sigma^2_{\text{input}} = g^\top \Sigma_z g
\]

Total predictive variance:

\
\sigma^2_{\text{total}} = \sigma^2_{\text{model}} + \sigma^2_{\text{input}}
\

**Outputs:**  
- Uncertainty plots  
- Gradient norm analysis  
- Per-feature variance attribution  
- Variance comparison histogram  

---

# 6. Hybrid Robust GPR Model (Physics + GPR Residual)

This is the most complete model:  
**Physics model** + **GPR-learned residual** + **full uncertainty propagation**.

### Steps:
1. Compute `jw_physical`  
2. Compute residual \( e = y - jw_\text{physical} \)  
3. Train GPR on residuals  
4. Hybrid prediction:
   \[
   y_\text{hybrid} = jw_\text{physical} + e_\text{GPR}
   \]
5. Compute hybrid gradients  
6. Propagate uncertainty using Delta method  
7. Validate using Monte-Carlo simulation  

**Outputs:**
- Hybrid parity plot  
- Physics vs Hybrid residual comparison  
- Total uncertainty vs model uncertainty  
- Analytical vs MC variance scatter plot  
- Worst-case variance mismatch diagnostics  

---

# 7. Monte-Carlo Validation

To confirm the correctness of Delta-method uncertainty propagation, Monte-Carlo simulation is performed:

1. Sample perturbed inputs  
2. Run hybrid model  
3. Compute empirical variance  
4. Compare with analytical Delta-method variance

**Provided diagnostics:**
- Scatter: Analytical vs MC variance  
- Relative error histogram  
- Standard deviation comparison  
- Worst mismatches reported  

---

# 8. Repository Structure

```
/src
    robust_gpr.py         # Full implementation of all models and workflows
/notebooks
    robust_gpr.ipynb      # Optional interactive notebook
/plots
    *.png                 # Generated parity, uncertainty, MC validation plots
/data
    results_final.csv     # Input dataset (optional)
/models
    *.joblib              # Trained ANN and GPR models
README.md
```

---

# 9. Running the Code

```
python robust_gpr.py
```

**Requirements:**
```
numpy
pandas
scikit-learn
scipy
matplotlib
tqdm
joblib
```

---

# 10. Summary

This repository provides a full end-to-end workflow for:

- Physical modeling of forward osmosis  
- Data-driven and hybrid surrogate modeling  
- Gaussian Process regression  
- Gradient-based uncertainty propagation  
- Monte-Carlo validation  
- Diagnostic visualization and analysis  

The implemented methods support accurate, explainable, and uncertainty-aware predictions suitable for design, simulation, or scientific investigation.

