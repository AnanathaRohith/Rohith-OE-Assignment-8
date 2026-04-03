import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Optimization Explorer", page_icon="🔬", layout="wide")

st.title("🔬 Optimization Explorer")

# Sidebar
page = st.sidebar.radio(
    "Choose Task",
    ["Task 1: Critical Points", "Task 2: Pareto Front", "Task 3: Optimization"]
)

# =========================================================
# TASK 1
# =========================================================
if page == "Task 1: Critical Points":

    st.header("📊 Critical Point Analysis")

    def f(x):
        return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + 0.25*x[4]**4 - 2*x[4]**2

    def hessian(x):
        H = np.zeros((5,5))
        for i in range(4):
            H[i,i] = 2
        H[4,4] = 3*x[4]**2 - 4
        return H

    def classify(x):
        eig = np.linalg.eigvalsh(hessian(x))
        if np.all(eig > 0):
            return "Local Minimum"
        elif np.all(eig < 0):
            return "Local Maximum"
        else:
            return "Saddle Point"

    critical = [
        np.array([0,0,0,0,0]),
        np.array([0,0,0,0,2]),
        np.array([0,0,0,0,-2])
    ]

    st.subheader("Known Critical Points")
    for cp in critical:
        st.write(cp, classify(cp), "f(x)=", f(cp))

    # NEW INTERACTIVE PART
    st.subheader("🧪 Try Your Own Point")
    x5 = st.slider("Choose x5", -5.0, 5.0, 0.0)
    user = np.array([0,0,0,0,x5])
    st.write("Point:", user)
    st.write("f(x):", round(f(user),3))
    st.write("Type:", classify(user))

    # Plot
    x = np.linspace(-4,4,200)
    y = 0.25*x**4 - 2*x**2

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='f(x5)'))

    fig.add_trace(go.Scatter(
        x=[x5], y=[f(user)],
        mode='markers',
        marker=dict(size=12, color='red'),
        name='Your Point'
    ))

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TASK 2
# =========================================================
elif page == "Task 2: Pareto Front":

    st.header("🎯 Pareto Optimization")

    data = {
        "ID":["L1","L2","L3","L4"],
        "Price":[1000,2000,1500,1200],
        "Performance":[70,90,80,75],
        "Battery":[10,6,12,11],
        "Weight":[1.5,2.5,1.3,1.4]
    }

    df = pd.DataFrame(data)

    # FILTERS
    st.sidebar.subheader("Filters")
    max_price = st.sidebar.slider("Max Price", 500, 3000, 3000)
    min_battery = st.sidebar.slider("Min Battery", 5, 15, 5)

    filtered = df[(df["Price"]<=max_price) & (df["Battery"]>=min_battery)]

    def pareto(i, data):
        p = data.iloc[i]
        for j, o in data.iterrows():
            if j==i: continue
            if (o.Price<=p.Price and o.Performance>=p.Performance and
                o.Battery>=p.Battery and o.Weight<=p.Weight and
                (o.Price<p.Price or o.Performance>p.Performance or
                 o.Battery>p.Battery or o.Weight<p.Weight)):
                return False
        return True

    filtered["Pareto"] = [pareto(i, filtered) for i in range(len(filtered))]

    st.dataframe(filtered)

    # Plot
    fig = px.scatter(filtered, x="Price", y="Performance",
                     color="Pareto", size="Battery")

    st.plotly_chart(fig, use_container_width=True)

    # Highlight
    choice = st.selectbox("Highlight Laptop", filtered["ID"])
    row = filtered[filtered["ID"]==choice].iloc[0]

    st.success(f"{choice} → Price:{row.Price}, Perf:{row.Performance}")

# =========================================================
# TASK 3
# =========================================================
elif page == "Task 3: Optimization":

    st.header("⚡ Optimization Algorithms")

    def rosen(x):
        return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

    st.subheader("🎯 Custom Start Point")

    x1 = st.number_input("x1", value=-1.0)
    x2 = st.number_input("x2", value=1.0)

    max_iter = st.slider("Max Iterations", 100, 5000, 1000)

    if st.button("Run Optimization"):

        x0 = np.array([x1,x2])
        path = [x0.copy()]

        def callback(xk):
            path.append(xk.copy())

        res = minimize(rosen, x0, method='CG',
                       callback=callback,
                       options={'maxiter':max_iter})

        path = np.array(path)

        # Contour
        x = np.linspace(-3,3,100)
        y = np.linspace(-3,3,100)
        X,Y = np.meshgrid(x,y)
        Z = (1-X)**2 + 100*(Y-X**2)**2

        fig = go.Figure()
        fig.add_trace(go.Contour(x=x,y=y,z=Z))

        fig.add_trace(go.Scatter(
            x=path[:,0], y=path[:,1],
            mode='lines+markers',
            line=dict(color='red')
        ))

        st.plotly_chart(fig, use_container_width=True)

        st.success(f"Final value: {res.fun:.5f}")

        # SIMPLE ANIMATION
        st.subheader("🎬 Step Visualization")
        for i in range(1,len(path)):
            fig = go.Figure()
            fig.add_trace(go.Contour(x=x,y=y,z=Z))
            fig.add_trace(go.Scatter(
                x=path[:i,0], y=path[:i,1],
                mode='lines+markers'
            ))
            st.plotly_chart(fig, use_container_width=True)
            time.sleep(0.05)
