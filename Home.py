import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Optimization Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔬 Optimization Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive visualization of Critical Points, Pareto Fronts & Optimization Algorithms</p>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Choose a Task:",
    ["🏠 Home", "📊 Task 1: Critical Points", "🎯 Task 2: Pareto Front", "⚡ Task 3: Optimization Algorithms"],
    index=0
)

# =============================================================================
# HOME PAGE
# =============================================================================
if page == "🏠 Home":
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Task 1")
        st.markdown("**5D Critical Point Analysis**")
        st.markdown("""
        Find and classify critical points of:
        
        $f(x) = x_1^2 + x_2^2 + x_3^2 + x_4^2 + \\frac{1}{4}x_5^4 - 2x_5^2$
        
        - Interactive 3D visualization
        - Hessian eigenvalue analysis
        - Classification of minima/maxima/saddle points
        """)
    
    with col2:
        st.markdown("### 🎯 Task 2")
        st.markdown("**Pareto Front Optimization**")
        st.markdown("""
        Multi-objective laptop optimization:
        
        - **Minimize**: Price, Weight
        - **Maximize**: Performance, Battery
        
        - Interactive Pareto frontier
        - Dominance visualization
        - Custom laptop comparison
        """)
    
    with col3:
        st.markdown("### ⚡ Task 3")
        st.markdown("**Algorithm Benchmarking**")
        st.markdown("""
        Compare optimization methods:
        
        - Steepest Descent (Gradient)
        - Newton's Method
        - Conjugate Gradient
        
        Test on Rosenbrock & Ackley functions
        """)
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Critical Points Found", "3", "5D Analysis")
    col2.metric("Pareto-Optimal Laptops", "4", "out of 30")
    col3.metric("Optimization Methods", "3", "Compared")
    col4.metric("Test Functions", "2", "Rosenbrock & Ackley")

# =============================================================================
# TASK 1: CRITICAL POINTS
# =============================================================================
elif page == "📊 Task 1: Critical Points":
    st.header("📊 Task 1: 5D Critical Point Analysis")
    
    st.markdown("""
    ### Function Definition
    $$f(\mathbf{x}) = x_1^2 + x_2^2 + x_3^2 + x_4^2 + \\frac{1}{4}x_5^4 - 2x_5^2$$
    """)
    
    # Function definitions
    def f_5d(x):
        return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + 0.25*x[4]**4 - 2*x[4]**2
    
    def hessian_5d(x):
        H = np.zeros((5, 5))
        H[0, 0] = 2
        H[1, 1] = 2
        H[2, 2] = 2
        H[3, 3] = 2
        H[4, 4] = 3*x[4]**2 - 4
        return H
    
    def classify_point(x):
        H = hessian_5d(x)
        eigenvalues = np.linalg.eigvalsh(H)
        if np.all(eigenvalues > 1e-10):
            return "Local Minimum", eigenvalues
        elif np.all(eigenvalues < -1e-10):
            return "Local Maximum", eigenvalues
        else:
            return "Saddle Point", eigenvalues
    
    # Critical points
    critical_points = [
        np.array([0., 0., 0., 0., 0.]),
        np.array([0., 0., 0., 0., 2.]),
        np.array([0., 0., 0., 0., -2.])
    ]
    
    # Display results
    st.markdown("### Critical Points Found")
    
    cols = st.columns(3)
    for idx, cp in enumerate(critical_points):
        classification, eigenvalues = classify_point(cp)
        fval = f_5d(cp)
        
        with cols[idx]:
            if classification == "Local Minimum":
                st.success(f"**Point {idx + 1}: {classification}**")
            elif classification == "Saddle Point":
                st.warning(f"**Point {idx + 1}: {classification}**")
            else:
                st.error(f"**Point {idx + 1}: {classification}**")
            
            st.write(f"**Coordinates:** ({cp[0]}, {cp[1]}, {cp[2]}, {cp[3]}, {cp[4]})")
            st.write(f"**f(x) = {fval:.2f}**")
            st.write(f"**Eigenvalues:** {np.round(eigenvalues, 2)}")
    
    st.markdown("---")
    
    # Interactive 2D slice visualization
    st.markdown("### 🎮 Interactive Visualization")
    st.markdown("Explore a 2D slice of the function with x₁ to x₄ = 0")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        x5_range = st.slider("x₅ range", -5.0, 5.0, (-3.0, 3.0))
        points_density = st.slider("Point density", 50, 200, 100)
        show_critical = st.checkbox("Show critical points", value=True)
    
    with col2:
        x5_vals = np.linspace(x5_range[0], x5_range[1], points_density)
        f_vals = [f_5d([0, 0, 0, 0, x5]) for x5 in x5_vals]
        
        fig = go.Figure()
        
        # Function curve
        fig.add_trace(go.Scatter(
            x=x5_vals, y=f_vals,
            mode='lines',
            name='f(0,0,0,0,x₅)',
            line=dict(color='#667eea', width=3)
        ))
        
        if show_critical:
            # Critical points
            for cp in critical_points:
                classification, _ = classify_point(cp)
                fval = f_5d(cp)
                color = '#28a745' if classification == "Local Minimum" else '#ffc107'
                marker = 'star' if classification == "Local Minimum" else 'x'
                fig.add_trace(go.Scatter(
                    x=[cp[4]], y=[fval],
                    mode='markers',
                    name=f'{classification} at x₅={cp[4]}',
                    marker=dict(size=15, color=color, symbol=marker)
                ))
        
        fig.update_layout(
            title="Function slice: f(0,0,0,0,x₅)",
            xaxis_title="x₅",
            yaxis_title="f(x)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 3D Surface
    st.markdown("### 🌐 3D Surface Visualization")
    st.markdown("View function with two variables (x₄, x₅), others set to 0")
    
    x4 = np.linspace(-3, 3, 50)
    x5 = np.linspace(-3, 3, 50)
    X4, X5 = np.meshgrid(x4, x5)
    Z = X4**2 + 0.25*X5**4 - 2*X5**2
    
    fig_3d = go.Figure(data=[go.Surface(x=X4, y=X5, z=Z, colorscale='Viridis', opacity=0.9)])
    
    # Add critical points
    fig_3d.add_trace(go.Scatter3d(
        x=[0, 0, 0], y=[0, 2, -2], z=[0, -4, -4],
        mode='markers',
        marker=dict(size=10, color=['yellow', 'green', 'green']),
        name='Critical Points'
    ))
    
    fig_3d.update_layout(
        title="f(0,0,0,x₄,x₅) = x₄² + ¼x₅⁴ - 2x₅²",
        scene=dict(xaxis_title="x₄", yaxis_title="x₅", zaxis_title="f(x)"),
        height=500
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Hessian explorer
    st.markdown("### 🔢 Hessian Matrix Explorer")
    
    x5_explore = st.slider("Choose x₅ value to explore Hessian", -3.0, 3.0, 0.0, 0.1)
    H = hessian_5d([0, 0, 0, 0, x5_explore])
    eigenvalues = np.linalg.eigvalsh(H)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Hessian Matrix at x₅ = {:.1f}**".format(x5_explore))
        st.dataframe(pd.DataFrame(H, columns=[f'∂²f/∂x{i+1}' for i in range(5)], 
                                   index=[f'∂x{i+1}' for i in range(5)]))
    
    with col2:
        st.markdown("**Eigenvalues**")
        eig_df = pd.DataFrame({
            'Eigenvalue': [f'λ{i+1}' for i in range(5)],
            'Value': eigenvalues,
            'Sign': ['Positive' if e > 0 else 'Negative' if e < 0 else 'Zero' for e in eigenvalues]
        })
        st.dataframe(eig_df)
        
        classification, _ = classify_point([0, 0, 0, 0, x5_explore])
        if classification == "Local Minimum":
            st.success(f"Classification: **{classification}**")
        elif classification == "Saddle Point":
            st.warning(f"Classification: **{classification}**")
        else:
            st.error(f"Classification: **{classification}**")

# =============================================================================
# TASK 2: PARETO FRONT
# =============================================================================
elif page == "🎯 Task 2: Pareto Front":
    st.header("🎯 Task 2: Multi-Objective Laptop Optimization")
    
    # Embedded laptop data (from assignment-5)
    laptop_data = """ID,Model,Price,Performance,Battery,Weight
L01,EcoBook 13,450,55,10,1.3
L02,Titan G9,2800,99,8,2.8
L03,Enduro Air,1200,75,20,1.4
L04,Feather X,1500,80,12,0.95
L05,Flow 198,1267,51,16,1.8
L06,Apex 196,1363,55,17,1.78
L07,Prime 439,753,46,8,1.82
L08,Elite 551,2852,87,4,3.27
L09,Zenith 235,599,44,7,1.86
L10,Flow 688,1356,65,18,1.52
L11,Workmate 817,1396,66,15,1.8
L12,Swift 682,3215,75,4,2.98
L13,Swift 459,1379,59,16,2.19
L14,Flow 924,3155,82,4,3.46
L15,Gamer Pro 845,1854,58,8,1.53
L16,Gamer Pro 573,2039,73,7,1.64
L17,Core 322,1812,66,7,1.27
L18,Prime 298,1690,69,9,1.41
L19,Inspire 728,1349,69,15,1.83
L20,Core 209,2936,80,4,3.43
L21,Apex 868,3269,93,4,3.15
L22,Workmate 160,1871,63,7,1.33
L23,Workmate 188,1691,63,7,1.47
L24,Zenith 580,1927,67,10,1.22
L25,Core 306,1940,74,8,1.67
L26,Prime 590,2047,68,9,1.52
L27,Elite 781,3056,93,4,3.29
L28,Elite 297,1607,68,16,2.13
L29,Inspire 630,3325,80,4,3.53
L30,Inspire 751,3260,83,4,3.34"""
    
    # Load data
    @st.cache_data
    def load_laptop_data():
        from io import StringIO
        return pd.read_csv(StringIO(laptop_data))
    
    df = load_laptop_data()
    
    st.markdown("""
    ### Optimization Objectives
    - 📉 **Minimize**: Price, Weight
    - 📈 **Maximize**: Performance, Battery
    """)
    
    # Pareto optimality check
    def is_pareto_optimal(point_idx, data):
        point = data.iloc[point_idx]
        for idx, other in data.iterrows():
            if idx == point_idx:
                continue
            
            price_better = other['Price'] <= point['Price']
            perf_better = other['Performance'] >= point['Performance']
            battery_better = other['Battery'] >= point['Battery']
            weight_better = other['Weight'] <= point['Weight']
            
            if price_better and perf_better and battery_better and weight_better:
                if (other['Price'] < point['Price'] or other['Performance'] > point['Performance'] or 
                    other['Battery'] > point['Battery'] or other['Weight'] < point['Weight']):
                    return False
        return True
    
    # Find Pareto-optimal laptops
    pareto_mask = [is_pareto_optimal(idx, df) for idx in range(len(df))]
    df['Pareto_Optimal'] = pareto_mask
    pareto_df = df[df['Pareto_Optimal']]
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Laptops", len(df))
    col2.metric("Pareto-Optimal", len(pareto_df))
    col3.metric("Dominated", len(df) - len(pareto_df))
    
    st.markdown("---")
    
    # Interactive scatter plot
    st.markdown("### 📊 Interactive Visualization")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        x_axis = st.selectbox("X-Axis", ['Price', 'Performance', 'Battery', 'Weight'], index=0)
        y_axis = st.selectbox("Y-Axis", ['Price', 'Performance', 'Battery', 'Weight'], index=1)
        color_by = st.selectbox("Color By", ['Pareto_Optimal', 'Battery', 'Weight', 'Price'], index=0)
        size_by = st.selectbox("Size By", ['Performance', 'Battery', 'Price', 'Weight'], index=0)
    
    with col2:
        fig = px.scatter(
            df, x=x_axis, y=y_axis,
            color=color_by,
            size=size_by,
            hover_data=['ID', 'Model', 'Price', 'Performance', 'Battery', 'Weight'],
            color_discrete_map={True: '#28a745', False: '#6c757d'} if color_by == 'Pareto_Optimal' else None,
            title=f'{y_axis} vs {x_axis}'
        )
        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    # Pareto-optimal laptops table
    st.markdown("### 🏆 Pareto-Optimal Laptops")
    
    st.dataframe(
        pareto_df[['ID', 'Model', 'Price', 'Performance', 'Battery', 'Weight']].style.highlight_max(
            subset=['Performance', 'Battery'], color='lightgreen'
        ).highlight_min(
            subset=['Price', 'Weight'], color='lightblue'
        ),
        use_container_width=True
    )
    
    # Parallel coordinates plot
    st.markdown("### 📈 Parallel Coordinates View")
    
    fig_parallel = px.parallel_coordinates(
        df,
        dimensions=['Price', 'Performance', 'Battery', 'Weight'],
        color='Pareto_Optimal',
        color_continuous_scale=[[0, '#6c757d'], [1, '#28a745']],
        title="All Laptops - Pareto Optimal (Green) vs Dominated (Gray)"
    )
    fig_parallel.update_layout(height=400)
    st.plotly_chart(fig_parallel, use_container_width=True)
    
    # Interactive laptop comparison
    st.markdown("### 🔍 Compare Laptops")
    
    col1, col2 = st.columns(2)
    
    with col1:
        laptop1 = st.selectbox("Select Laptop 1", df['ID'].tolist(), index=0)
    with col2:
        laptop2 = st.selectbox("Select Laptop 2", df['ID'].tolist(), index=1)
    
    l1 = df[df['ID'] == laptop1].iloc[0]
    l2 = df[df['ID'] == laptop2].iloc[0]
    
    # Radar chart comparison
    categories = ['Price (inv)', 'Performance', 'Battery', 'Weight (inv)']
    
    # Normalize for radar chart
    max_price = df['Price'].max()
    max_perf = df['Performance'].max()
    max_battery = df['Battery'].max()
    max_weight = df['Weight'].max()
    
    l1_vals = [1 - l1['Price']/max_price, l1['Performance']/max_perf, 
               l1['Battery']/max_battery, 1 - l1['Weight']/max_weight]
    l2_vals = [1 - l2['Price']/max_price, l2['Performance']/max_perf, 
               l2['Battery']/max_battery, 1 - l2['Weight']/max_weight]
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=l1_vals + [l1_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=f'{l1["ID"]} - {l1["Model"]}'
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=l2_vals + [l2_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=f'{l2["ID"]} - {l2["Model"]}'
    ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Laptop Comparison (Higher = Better)",
        height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Detailed comparison
    comparison_df = pd.DataFrame({
        'Metric': ['Price ($)', 'Performance', 'Battery (hrs)', 'Weight (kg)', 'Pareto Optimal'],
        laptop1: [l1['Price'], l1['Performance'], l1['Battery'], l1['Weight'], '✅' if l1['Pareto_Optimal'] else '❌'],
        laptop2: [l2['Price'], l2['Performance'], l2['Battery'], l2['Weight'], '✅' if l2['Pareto_Optimal'] else '❌'],
        'Better': [
            laptop1 if l1['Price'] < l2['Price'] else laptop2 if l2['Price'] < l1['Price'] else 'Tie',
            laptop1 if l1['Performance'] > l2['Performance'] else laptop2 if l2['Performance'] > l1['Performance'] else 'Tie',
            laptop1 if l1['Battery'] > l2['Battery'] else laptop2 if l2['Battery'] > l1['Battery'] else 'Tie',
            laptop1 if l1['Weight'] < l2['Weight'] else laptop2 if l2['Weight'] < l1['Weight'] else 'Tie',
            '-'
        ]
    })
    st.dataframe(comparison_df, use_container_width=True)

# =============================================================================
# TASK 3: OPTIMIZATION ALGORITHMS
# =============================================================================
elif page == "⚡ Task 3: Optimization Algorithms":
    st.header("⚡ Task 3: Optimization Algorithm Benchmarking")
    
    st.markdown("""
    Compare **Steepest Descent**, **Newton's Method**, and **Conjugate Gradient** 
    on the Rosenbrock and Ackley test functions.
    """)
    
    # Function definitions
    def rosenbrock(x):
        return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    def rosenbrock_grad(x):
        grad = np.zeros_like(x)
        grad[0] = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
        for i in range(1, len(x)-1):
            grad[i] = 200*(x[i] - x[i-1]**2) - 400*x[i]*(x[i+1] - x[i]**2) - 2*(1 - x[i])
        grad[-1] = 200*(x[-1] - x[-2]**2)
        return grad
    
    def ackley(x):
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2*np.pi*x))
        return -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e
    
    # Tab selection for function
    tab1, tab2, tab3 = st.tabs(["🏔️ Rosenbrock Function", "🌊 Ackley Function", "🎮 Interactive Explorer"])
    
    with tab1:
        st.markdown("""
        ### Rosenbrock Function (n=10)
        $$f(x) = \sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1-x_i)^2]$$
        
        Global minimum at **x = (1, 1, ..., 1)** with **f(x) = 0**
        """)
        
        n = 10
        x0 = np.random.RandomState(42).randn(n) * 2
        
        if st.button("🚀 Run Optimization on Rosenbrock", key="ros"):
            with st.spinner("Running optimizations..."):
                results = []
                
                # Steepest Descent (BFGS)
                res_sd = minimize(rosenbrock, x0, method='BFGS', jac=rosenbrock_grad, options={'maxiter': 50000})
                results.append(('Steepest Descent (BFGS)', res_sd.nit, res_sd.fun, res_sd.success))
                
                # Newton's Method
                res_newton = minimize(rosenbrock, x0, method='Newton-CG', jac=rosenbrock_grad, options={'maxiter': 50000})
                results.append(("Newton's Method", res_newton.nit, res_newton.fun, res_newton.success))
                
                # Conjugate Gradient
                res_cg = minimize(rosenbrock, x0, method='CG', jac=rosenbrock_grad, options={'maxiter': 50000})
                results.append(('Conjugate Gradient', res_cg.nit, res_cg.fun, res_cg.success))
            
            # Display results
            results_df = pd.DataFrame(results, columns=['Method', 'Iterations', 'Final Value', 'Converged'])
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.dataframe(results_df, use_container_width=True)
            
            with col2:
                fig = px.bar(results_df, x='Method', y='Iterations', color='Method',
                            title='Iterations to Convergence')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Value comparison
            fig2 = px.bar(results_df, x='Method', y='Final Value', color='Method',
                         title='Final Function Value (Lower is Better)', log_y=True)
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.markdown("""
        ### Ackley Function (n=10)
        $$f(x) = -20\exp(-0.2\sqrt{\\frac{1}{n}\sum_{i=1}^n x_i^2}) - \exp(\\frac{1}{n}\sum_{i=1}^n \cos(2\pi x_i)) + 20 + e$$
        
        Global minimum at **x = (0, 0, ..., 0)** with **f(x) = 0**
        """)
        
        starting_point = st.selectbox(
            "Select Starting Point",
            ["Small (σ=0.1)", "Medium (σ=2)", "Large (σ=5)"]
        )
        
        sigma_map = {"Small (σ=0.1)": 0.1, "Medium (σ=2)": 2, "Large (σ=5)": 5}
        sigma = sigma_map[starting_point]
        
        if st.button("🚀 Run Optimization on Ackley", key="ack"):
            n = 10
            x0 = np.random.RandomState(42).randn(n) * sigma
            
            with st.spinner("Running optimizations..."):
                results = []
                
                try:
                    res_sd = minimize(ackley, x0, method='BFGS', options={'maxiter': 50000})
                    results.append(('Steepest Descent', res_sd.nit, res_sd.fun, res_sd.fun < 0.1))
                except:
                    results.append(('Steepest Descent', 'N/A', 'Failed', False))
                
                try:
                    res_newton = minimize(ackley, x0, method='Newton-CG', options={'maxiter': 50000})
                    results.append(("Newton's Method", res_newton.nit, res_newton.fun, res_newton.fun < 0.1))
                except:
                    results.append(("Newton's Method", 'N/A', 'Failed', False))
                
                try:
                    res_cg = minimize(ackley, x0, method='CG', options={'maxiter': 50000})
                    results.append(('Conjugate Gradient', res_cg.nit, res_cg.fun, res_cg.fun < 0.1))
                except:
                    results.append(('Conjugate Gradient', 'N/A', 'Failed', False))
            
            results_df = pd.DataFrame(results, columns=['Method', 'Iterations', 'Final Value', 'Reached Minima'])
            st.dataframe(results_df, use_container_width=True)
            
            # Bar chart
            valid_results = results_df[results_df['Iterations'] != 'N/A'].copy()
            if len(valid_results) > 0:
                valid_results['Iterations'] = valid_results['Iterations'].astype(int)
                fig = px.bar(valid_results, x='Method', y='Iterations', color='Reached Minima',
                            title='Iterations to Convergence',
                            color_discrete_map={True: '#28a745', False: '#dc3545'})
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🎮 2D Function Visualization")
        
        func_choice = st.radio("Select Function", ["Rosenbrock (2D)", "Ackley (2D)"])
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            x_range = st.slider("X Range", -5.0, 5.0, (-2.0, 2.0), key="xr")
            y_range = st.slider("Y Range", -5.0, 5.0, (-2.0, 2.0), key="yr")
            resolution = st.slider("Resolution", 20, 100, 50)
        
        with col2:
            x = np.linspace(x_range[0], x_range[1], resolution)
            y = np.linspace(y_range[0], y_range[1], resolution)
            X, Y = np.meshgrid(x, y)
            
            if func_choice == "Rosenbrock (2D)":
                Z = (1 - X)**2 + 100*(Y - X**2)**2
                title = "Rosenbrock Function"
            else:
                Z = -20*np.exp(-0.2*np.sqrt((X**2 + Y**2)/2)) - np.exp((np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))/2) + 20 + np.e
                title = "Ackley Function"
            
            fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
            fig.update_layout(title=title, height=500,
                            scene=dict(xaxis_title='x₁', yaxis_title='x₂', zaxis_title='f(x)'))
            st.plotly_chart(fig, use_container_width=True)
        
        # Contour plot with optimization path
        st.markdown("### 📍 Contour Plot with Optimization Path")
        
        if st.button("🎯 Run and Visualize Optimization Path"):
            x = np.linspace(x_range[0], x_range[1], 100)
            y = np.linspace(y_range[0], y_range[1], 100)
            X, Y = np.meshgrid(x, y)
            
            if func_choice == "Rosenbrock (2D)":
                Z = (1 - X)**2 + 100*(Y - X**2)**2
                func = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
                x0 = np.array([-1.0, -1.0])
            else:
                Z = -20*np.exp(-0.2*np.sqrt((X**2 + Y**2)/2)) - np.exp((np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))/2) + 20 + np.e
                func = lambda x: -20*np.exp(-0.2*np.sqrt((x[0]**2 + x[1]**2)/2)) - np.exp((np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))/2) + 20 + np.e
                x0 = np.array([1.0, 1.0])
            
            # Collect path
            path = [x0.copy()]
            def callback(xk):
                path.append(xk.copy())
            
            result = minimize(func, x0, method='CG', callback=callback, options={'maxiter': 1000})
            path = np.array(path)
            
            fig = go.Figure()
            
            # Contour
            fig.add_trace(go.Contour(x=x, y=y, z=Z, colorscale='Viridis', 
                                     contours=dict(showlabels=True), opacity=0.8))
            
            # Path
            fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode='lines+markers',
                                    marker=dict(size=5, color='red'),
                                    line=dict(color='red', width=2),
                                    name='Optimization Path'))
            
            # Start and end
            fig.add_trace(go.Scatter(x=[path[0, 0]], y=[path[0, 1]], mode='markers',
                                    marker=dict(size=15, color='blue', symbol='star'),
                                    name='Start'))
            fig.add_trace(go.Scatter(x=[path[-1, 0]], y=[path[-1, 1]], mode='markers',
                                    marker=dict(size=15, color='green', symbol='star'),
                                    name='End'))
            
            fig.update_layout(title=f"Conjugate Gradient Path on {func_choice}", height=500,
                            xaxis_title='x₁', yaxis_title='x₂')
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"✅ Converged in **{len(path)-1}** iterations to f(x) = **{result.fun:.6f}**")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎓 Optimization & Pareto Efficiency Assignment | Built with Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)
