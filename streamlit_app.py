"""
Streamlit App for Price Optimization Score Breakdown

Students upload their submission CSV and see detailed score breakdown.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import os
from evaluation_breakdown import simulate_with_breakdown

# Configuration
DATA_DIR = "new_dataset"  # Change this for deployment
SHOW_DETAILED_BREAKDOWN = os.getenv("SHOW_DETAILED_BREAKDOWN", "false").lower() == "true"

# Page config
st.set_page_config(
    page_title="Price Optimization Score Checker",
    page_icon="📊",
    layout="wide",
)

# Title
st.title("📊 Price Optimization Score Breakdown")
st.markdown("Upload your submission CSV to see your score breakdown.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. Upload your submission CSV file
    2. Wait for evaluation (~20 seconds)
    3. Review your score breakdown

    **Required Columns:**
    - ID
    - store_id
    - sku_id
    - date (DD/MM/YYYY)
    - proposed_price

    **File Format:**
    - 33,600 rows (20 stores × 120 SKUs × 14 days)
    - CSV format
    """)

    st.markdown("---")
    st.markdown("**Competition Rules:**")
    st.markdown("""
    - Price ≥ Cost × (1 + VAT)
    - Discount ≤ 30% off regular price
    - Price endings: .0, .5, or .9
    - Minimize price changes
    """)

# File uploader
uploaded_file = st.file_uploader(
    "Upload your submission CSV",
    type=['csv'],
    help="Upload a CSV file with your price predictions"
)

if uploaded_file is not None:
    try:
        # Load submission
        with st.spinner("Loading submission..."):
            submission = pd.read_csv(uploaded_file)

        # Validate columns
        required_cols = ['ID', 'store_id', 'sku_id', 'date', 'proposed_price']
        missing_cols = [col for col in required_cols if col not in submission.columns]

        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.stop()

        # Show preview
        with st.expander("📄 Preview Submission (first 10 rows)"):
            st.dataframe(submission.head(10))

        # Validate row count
        expected_rows = 33600
        if len(submission) != expected_rows:
            st.warning(f"⚠️ Expected {expected_rows:,} rows, got {len(submission):,} rows")

        # Validate data types and convert date
        try:
            submission['store_id'] = submission['store_id'].astype(int)
            submission['sku_id'] = submission['sku_id'].astype(int)
            submission['proposed_price'] = submission['proposed_price'].astype(float)

            # Convert date format from DD/MM/YYYY to YYYY-MM-DD
            if "/" in str(submission['date'].iloc[0]):
                submission['date'] = submission['date'].apply(
                    lambda x: datetime.strptime(str(x), "%d/%m/%Y").strftime("%Y-%m-%d")
                )
        except Exception as e:
            st.error(f"❌ Data validation error: {e}")
            st.stop()

        # Check for invalid prices
        if (submission['proposed_price'] <= 0).any():
            st.error("❌ All proposed prices must be greater than 0")
            st.stop()

        # Run evaluation
        st.markdown("---")
        st.subheader("🔄 Running Evaluation...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("Evaluating submission... This may take ~20 seconds"):
            status_text.text("Processing 14 days of simulation...")
            progress_bar.progress(30)

            # Run simulation
            result = simulate_with_breakdown(submission, DATA_DIR)

            progress_bar.progress(100)
            status_text.text("✓ Evaluation complete!")

        # Display results
        st.markdown("---")
        st.success("✅ Evaluation Complete!")

        # Main metrics - ALWAYS SHOWN
        st.subheader("📈 Overall Score Dashboard")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                label="🎯 Final Score",
                value=f"{result['final_score']:,.0f}",
                delta="Higher is better"
            )

        with col2:
            st.metric(
                label="💰 Total Profit",
                value=f"฿{result['total_profit']:,.0f}",
            )

        with col3:
            st.metric(
                label="⚠️ Stockout Penalty",
                value=f"-฿{result['stockout_penalty']:,.0f}",
            )

        with col4:
            st.metric(
                label="❌ Violations",
                value=f"{result['violations']:,}",
                delta="Lower is better"
            )

        with col5:
            st.metric(
                label="📊 Instability",
                value=f"{result['instability']:,}",
                delta="Lower is better"
            )

        # Score interpretation
        st.markdown("---")
        st.subheader("💡 Score Interpretation")

        if result['final_score'] > 5000000:
            st.success("🎉 Excellent! Your score beats the baseline!")
        elif result['final_score'] > 0:
            st.info("👍 Good! Your score is positive. Try to optimize further.")
        else:
            st.warning("⚠️ Your score is negative. Check for violations and stockouts.")

        # Detailed breakdown (controlled by environment variable)
        if SHOW_DETAILED_BREAKDOWN:
            st.markdown("---")
            st.subheader("📊 Detailed Breakdown")

            # Daily metrics chart
            st.markdown("### Daily Performance")
            daily_df = result['daily_metrics']

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_df['date'],
                y=daily_df['profit'],
                mode='lines+markers',
                name='Daily Profit',
                line=dict(color='green', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=daily_df['date'],
                y=daily_df['stockout_penalty'],
                mode='lines+markers',
                name='Daily Stockout Penalty',
                line=dict(color='red', width=2)
            ))

            fig.update_layout(
                title="Profit and Stockout Penalty Over 14 Days",
                xaxis_title="Date",
                yaxis_title="Amount ($)",
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Violation details
            if result['violations'] > 0:
                st.markdown("### Violation Details")
                violations_df = pd.DataFrame(result['violation_details'])

                # Group by type
                violation_counts = violations_df['type'].value_counts()

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown("**Violations by Type:**")
                    for vtype, count in violation_counts.items():
                        st.write(f"- {vtype}: {count}")

                with col2:
                    # Show sample violations
                    st.markdown("**Sample Violations (first 10):**")
                    st.dataframe(violations_df.head(10))

        # Comparison with baseline
        st.markdown("---")
        st.subheader("🏆 Comparison with Baseline")

        baseline_score = 5241817.20
        diff = result['final_score'] - baseline_score
        diff_pct = (diff / baseline_score) * 100

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Your Score", f"{result['final_score']:,.0f}")

        with col2:
            st.metric("Baseline Score", f"{baseline_score:,.0f}")

        with col3:
            st.metric(
                "Difference",
                f"{diff:,.0f}",
                delta=f"{diff_pct:+.1f}%"
            )

        # Download results
        st.markdown("---")
        st.subheader("💾 Download Results")

        # Create summary CSV
        summary = pd.DataFrame([{
            'Final Score': result['final_score'],
            'Total Profit': result['total_profit'],
            'Stockout Penalty': result['stockout_penalty'],
            'Violations': result['violations'],
            'Instability': result['instability'],
        }])

        csv = summary.to_csv(index=False)

        st.download_button(
            label="Download Score Summary",
            data=csv,
            file_name="score_summary.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error processing submission: {str(e)}")
        st.exception(e)

else:
    # Show instructions when no file uploaded
    st.info("👆 Upload your submission CSV file to get started!")

    # Show example format
    st.markdown("### 📝 Example Submission Format")

    example_data = {
        'ID': [1, 2, 3, 4, 5],
        'store_id': [1, 1, 1, 1, 1],
        'sku_id': ["100000", "100001", "100002", "100003", "100004"],
        'date': ['14/08/2025', '14/08/2025', '14/08/2025', '14/08/2025', '14/08/2025'],
        'proposed_price': [10, 20, 30, 33, 44]
    }

    st.dataframe(pd.DataFrame(example_data))

    st.markdown("""
    **Tips for a good score:**
    - Keep prices stable (avoid frequent changes)
    - Ensure prices are above cost
    - Don't discount more than 30%
    - Use valid price endings (.0, .5, .9)
    - Balance profit vs stockouts
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Price Optimization Competition Score Checker</p>
        <p style='font-size: 0.8em; color: gray;'>
            Evaluation time: ~20 seconds | Dataset: 14-day forecast period
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
