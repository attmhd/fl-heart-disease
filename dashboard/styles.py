import streamlit as st


def apply_custom_styling():
    st.markdown(
        """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            max-width: 100%;
            padding: 0 1rem;
        }

        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: #000000;
            margin-bottom: 0.5rem;
            margin-top: -3.5rem;
            letter-spacing: -0.02em;
        }

        .sub-title {
            font-size: 1.1rem;
            color: #64748b;
            margin-bottom: 3rem;
            font-weight: 400;
        }

        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #000000;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #1e293b;
        }

        .metric-card {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 16px;
            padding: 1.5rem;
            transition: transform 0.2s ease;
        }

        .metric-card:hover {
            border-color: #3b82f6;
        }

        .metric-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: #f8fafc;
        }

        .metric-label {
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 0.5rem;
        }

        .status-healthy {
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #10b981;
            padding: 0.4rem 1.2rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .status-offline {
            background-color: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #ef4444;
            padding: 0.4rem 1.2rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
            font-size: 0.9rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .stTabs [aria-selected="true"] {
            color: #3b82f6 !important;
        }
        div[data-testid="stVerticalBlock"] > div:first-child {
            padding-top: 0;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown('<div class="main-title">Heart Disease Federated Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interactive Blackbox Testing & Centralized Telemetry Demo Portal</div>', unsafe_allow_html=True)
