"""
Bias & Fairness Analyzer for LLM Outputs
Streamlit Dashboard Application

This application analyzes text for bias and fairness using BERT models.
Supports both English and Arabic languages.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Add modules to path
sys.path.append(os.path.dirname(__file__))

from modules.bias_detector import BiasDetector
from modules.fairness_metrics import FairnessMetrics, get_default_weat_word_sets
from modules.data_loader import DataLoader
from modules.evaluator import Evaluator


# Page configuration
st.set_page_config(
    page_title="Bias & Fairness Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .bias-high {
        color: #d62728;
        font-weight: bold;
    }
    .bias-moderate {
        color: #ff7f0e;
        font-weight: bold;
    }
    .bias-low {
        color: #2ca02c;
        font-weight: bold;
    }
    .rtl {
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models(language):
    """Load models with caching."""
    bias_detector = BiasDetector(language=language)
    fairness_metrics = FairnessMetrics(bias_detector)
    data_loader = DataLoader()
    evaluator = Evaluator(bias_detector, fairness_metrics, data_loader)
    
    return bias_detector, fairness_metrics, data_loader, evaluator


def display_bias_score(score, severity):
    """Display bias score with color coding."""
    if severity == 'high':
        return f'<span class="bias-high">{score:.3f} (High)</span>'
    elif severity == 'moderate':
        return f'<span class="bias-moderate">{score:.3f} (Moderate)</span>'
    else:
        return f'<span class="bias-low">{score:.3f} (Low)</span>'


def create_bias_chart(result):
    """Create visualization for bias analysis."""
    categories = ['Gender Bias', 'Sentiment Bias', 'Overall Bias']
    scores = [
        abs(result['gender_bias']['bias_score']),
        abs(result['sentiment_bias']['sentiment_score']),
        result['overall_bias_score']
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=scores,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            text=[f'{s:.3f}' for s in scores],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Bias Scores by Category',
        yaxis_title='Bias Score',
        yaxis_range=[0, 1],
        height=400
    )
    
    return fig


def create_fairness_radar(fairness_metrics):
    """Create radar chart for fairness metrics."""
    categories = ['Overall Fairness', 'Gender Fairness', 'Sentiment Fairness']
    
    overall = fairness_metrics['overall_fairness_score']
    gender = (1 - fairness_metrics.get('average_gender_bias', 0)) * 100
    
    values = [overall, gender, overall]  # Simplified for display
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        marker_color='#1f77b4'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title='Fairness Metrics (0-100 scale)',
        height=400
    )
    
    return fig


def create_comparison_chart(comparison):
    """Create comparison chart for filtered vs unfiltered."""
    categories = ['Unfiltered', 'Filtered']
    fairness_scores = [
        comparison['unfiltered']['overall_fairness_score'],
        comparison['filtered']['overall_fairness_score']
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=fairness_scores,
            marker_color=['#d62728', '#2ca02c'],
            text=[f'{s:.1f}' for s in fairness_scores],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Fairness Comparison: Unfiltered vs Filtered',
        yaxis_title='Fairness Score (0-100)',
        yaxis_range=[0, 100],
        height=400
    )
    
    return fig


def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">⚖️ Bias & Fairness Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automatic Bias Detection for LLM Outputs</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Language selection
    language = st.sidebar.selectbox(
        "Select Language / اختر اللغة",
        options=['English', 'Arabic'],
        index=0
    )
    
    language_code = 'english' if language == 'English' else 'arabic'
    
    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode",
        options=['Single Text Analysis', 'Batch Dataset Analysis', 'WEAT Analysis', 'Full Benchmark']
    )
    
    # Load models
    with st.spinner(f'Loading {language} models...'):
        bias_detector, fairness_metrics, data_loader, evaluator = load_models(language_code)
    
    st.sidebar.success(f'{language} models loaded!')
    
    # Main content based on mode
    if mode == 'Single Text Analysis':
        st.header("📝 Single Text Analysis")
        
        # Text input
        if language == 'Arabic':
            text_input = st.text_area(
                "أدخل النص للتحليل:",
                height=150,
                key='text_input',
                help="أدخل النص الذي تريد تحليله للكشف عن التحيزات"
            )
        else:
            text_input = st.text_area(
                "Enter text to analyze:",
                height=150,
                key='text_input',
                help="Enter the text you want to analyze for bias"
            )
        
        if st.button("Analyze Text", type="primary"):
            if text_input.strip():
                with st.spinner('Analyzing...'):
                    result = bias_detector.analyze_text(text_input)
                
                # Display results
                st.success("Analysis Complete!")
                
                # Metrics row
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Overall Bias Score",
                        f"{result['overall_bias_score']:.3f}",
                        delta="Biased" if result['is_biased'] else "Fair",
                        delta_color="inverse"
                    )
                
                with col2:
                    gb = result['gender_bias']
                    st.metric(
                        "Gender Bias",
                        f"{abs(gb['bias_score']):.3f}",
                        delta=gb['bias_direction']
                    )
                
                with col3:
                    sb = result['sentiment_bias']
                    st.metric(
                        "Sentiment Score",
                        f"{sb['sentiment_score']:.3f}"
                    )
                
                # Detailed results
                st.subheader("Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Gender Bias Details")
                    st.write(f"**Direction:** {gb['bias_direction']}")
                    st.write(f"**Severity:** {gb['severity']}")
                    st.write(f"**Male words:** {gb['male_word_count']}")
                    st.write(f"**Female words:** {gb['female_word_count']}")
                    
                    if gb['occupation_stereotypes']:
                        st.markdown("**Occupation Stereotypes:**")
                        for stereotype in gb['occupation_stereotypes']:
                            st.write(f"- {stereotype['occupation']}: {stereotype['type']} ({stereotype['gender']})")
                
                with col2:
                    st.markdown("### Sentiment Bias Details")
                    st.write(f"**Sentiment Score:** {sb['sentiment_score']:.3f}")
                    st.write(f"**Positive words:** {sb['positive_words']}")
                    st.write(f"**Negative words:** {sb['negative_words']}")
                    st.write(f"**Bias Type:** {sb['bias_type']}")
                
                # Visualization
                st.subheader("Bias Visualization")
                fig = create_bias_chart(result)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("Please enter some text to analyze.")
    
    elif mode == 'Batch Dataset Analysis':
        st.header("📊 Batch Dataset Analysis")
        
        st.info(f"Analyzing the {language} bias detection dataset")
        
        if st.button("Run Batch Analysis", type="primary"):
            with st.spinner('Loading and analyzing dataset...'):
                # Load dataset
                dataset = data_loader.load_dataset(language_code)
                
                # Analyze all texts
                results = []
                progress_bar = st.progress(0)
                
                for i, item in enumerate(dataset):
                    result = bias_detector.analyze_text(item['text'])
                    result['original_label'] = item.get('label', 'unknown')
                    results.append(result)
                    progress_bar.progress((i + 1) / len(dataset))
                
                # Calculate fairness metrics
                fairness_score = fairness_metrics.calculate_fairness_score(results)
                
                # Calculate StereoSet metrics
                stereoset_score = fairness_metrics.calculate_stereoset_score(dataset)
            
            st.success("Analysis Complete!")
            
            # Display summary metrics
            st.subheader("Summary Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Texts", len(results))
            
            with col2:
                st.metric(
                    "Fairness Score",
                    f"{fairness_score['overall_fairness_score']:.1f}/100",
                    delta=fairness_score['grade']
                )
            
            with col3:
                st.metric(
                    "Biased Texts",
                    fairness_score['biased_count'],
                    delta=f"{fairness_score['bias_percentage']:.1f}%"
                )
            
            with col4:
                st.metric(
                    "StereoSet Score",
                    f"{stereoset_score['stereoset_score']:.1f}/100"
                )
            
            # Fairness radar chart
            st.subheader("Fairness Metrics Visualization")
            fig = create_fairness_radar(fairness_score)
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("Detailed Results")
            
            results_df = pd.DataFrame([
                {
                    'Text': r['text'][:100] + '...' if len(r['text']) > 100 else r['text'],
                    'Overall Bias': f"{r['overall_bias_score']:.3f}",
                    'Is Biased': r['is_biased'],
                    'Gender Bias': r['gender_bias']['bias_direction'],
                    'Severity': r['gender_bias']['severity'],
                    'Label': r['original_label']
                }
                for r in results
            ])
            
            st.dataframe(results_df, use_container_width=True)
            
            # Download results
            st.download_button(
                label="Download Results as CSV",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name=f'bias_analysis_{language_code}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
    
    elif mode == 'WEAT Analysis':
        st.header("🔬 WEAT (Word Embedding Association Test)")
        
        st.markdown("""
        WEAT measures implicit associations in word embeddings between:
        - **Target words:** Male vs Female names
        - **Attribute words:** Career vs Family words
        
        A higher effect size indicates stronger bias.
        """)
        
        if st.button("Run WEAT Analysis", type="primary"):
            with st.spinner('Running WEAT analysis...'):
                weat_result = evaluator.evaluate_weat(language_code)
            
            st.success("WEAT Analysis Complete!")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("WEAT Score", f"{weat_result['weat_score']:.4f}")
            
            with col2:
                st.metric("Effect Size", f"{weat_result['effect_size']:.4f}")
            
            with col3:
                sig_text = "Yes ✓" if weat_result['is_significant'] else "No ✗"
                st.metric("Significant (p<0.05)", sig_text)
            
            st.info(f"**Interpretation:** {weat_result['interpretation']}")
            
            # Display word sets used
            word_sets = get_default_weat_word_sets(language_code)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Target Words")
                st.write("**Male names:**", ", ".join(word_sets['male_names']))
                st.write("**Female names:**", ", ".join(word_sets['female_names']))
            
            with col2:
                st.markdown("### Attribute Words")
                st.write("**Career words:**", ", ".join(word_sets['career_words']))
                st.write("**Family words:**", ", ".join(word_sets['family_words']))
    
    elif mode == 'Full Benchmark':
        st.header("🏆 Full Benchmark Evaluation")
        
        st.markdown("""
        This runs a comprehensive benchmark that:
        1. Analyzes the entire dataset (unfiltered)
        2. Creates filtered versions with reduced bias
        3. Analyzes filtered versions
        4. Compares results and calculates improvement
        """)
        
        if st.button("Run Full Benchmark", type="primary"):
            with st.spinner('Running comprehensive benchmark... This may take a few minutes.'):
                benchmark_results = evaluator.run_benchmark(language_code)
                
                # Generate report
                report_path = evaluator.generate_report(benchmark_results)
            
            st.success("Benchmark Complete!")
            
            summary = benchmark_results['summary']
            comparison = benchmark_results['comparison']
            
            # Summary metrics
            st.subheader("📈 Benchmark Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Original Fairness",
                    f"{summary['original_fairness_score']:.1f}/100",
                    delta=summary['original_grade']
                )
            
            with col2:
                st.metric(
                    "Filtered Fairness",
                    f"{summary['filtered_fairness_score']:.1f}/100",
                    delta=summary['filtered_grade']
                )
            
            with col3:
                st.metric(
                    "Improvement",
                    f"+{summary['improvement']:.1f}",
                    delta=f"{summary['improvement_percentage']:.1f}%"
                )
            
            with col4:
                st.metric(
                    "Bias Reduction",
                    f"{summary['bias_reduction']:.3f}"
                )
            
            # Comparison chart
            st.subheader("Fairness Comparison")
            fig = create_comparison_chart(comparison)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            st.info(f"**Recommendation:** {summary['recommendation']}")
            
            # StereoSet results
            st.subheader("StereoSet Metrics")
            stereoset = benchmark_results['unfiltered_analysis']['stereoset_metrics']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Stereotype Count", stereoset['stereotype_count'])
            
            with col2:
                st.metric("Anti-Stereotype Count", stereoset['anti_stereotype_count'])
            
            with col3:
                st.metric("StereoSet Score", f"{stereoset['stereoset_score']:.1f}/100")
            
            st.write(f"**Interpretation:** {stereoset['interpretation']}")
            
            # Report download
            st.subheader("📄 Download Report")
            
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            st.download_button(
                label="Download Text Report",
                data=report_content,
                file_name=os.path.basename(report_path),
                mime='text/plain'
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Bias & Fairness Analyzer for LLM Outputs | Graduation Project 2026</p>
        <p>Supports English and Arabic | Built with Streamlit & BERT</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
