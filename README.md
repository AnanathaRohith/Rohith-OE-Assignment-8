# 🔬 Optimization Explorer

An interactive Streamlit app for exploring optimization concepts including critical points, Pareto fronts, and optimization algorithms.

## Features

### 📊 Task 1: 5D Critical Point Analysis
- Interactive visualization of the 5D function: f(x) = x₁² + x₂² + x₃² + x₄² + ¼x₅⁴ - 2x₅²
- 3D surface plots
- Hessian matrix explorer with eigenvalue analysis
- Classification of critical points (minima, maxima, saddle points)

### 🎯 Task 2: Pareto Front Optimization
- Multi-objective laptop optimization (Price, Performance, Battery, Weight)
- Interactive scatter plots with customizable axes
- Parallel coordinates visualization
- Radar chart laptop comparison
- Pareto-optimal identification

### ⚡ Task 3: Optimization Algorithms
- Compare Steepest Descent, Newton's Method, and Conjugate Gradient
- Test on Rosenbrock and Ackley functions
- 3D surface visualizations
- Animated optimization paths on contour plots

## 🚀 Deployment to Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your repo, branch = `main`, main file = `Home.py`
5. Click **Deploy**

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
streamlit run Home.py
```

## 📁 Files

- `Home.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `README.md` - This file

## 🛠️ Technologies

- **Streamlit** - Web framework
- **Plotly** - Interactive visualizations
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **SciPy** - Optimization algorithms
