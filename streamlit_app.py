import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import json
import os

# استيراد الوحدات المخصصة
try:
    from utils.text_classifier import NewsClassifier
    from utils.text_summarizer import TextSummarizer
    from utils.entity_extractor import EntityExtractor
except ImportError as e:
    st.error(f"خطأ في استيراد الوحدات: {e}")
    st.stop()

# إعدادات الصفحة
st.set_page_config(
    page_title="محلل الأخبار الذكي",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تحميل CSS مخصص
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 2rem;
    }
    
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1E88E5;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# تهيئة الجلسة
@st.cache_resource
def initialize_models():
    """تحميل جميع النماذج"""
    try:
        classifier = NewsClassifier()
        summarizer = TextSummarizer()
        extractor = EntityExtractor()
        return classifier, summarizer, extractor
    except Exception as e:
        st.error(f"فشل في تحميل النماذج: {e}")
        return None, None, None

# تحميل النماذج
classifier, summarizer, extractor = initialize_models()

# العنوان الرئيسي
st.markdown('<h1 class="main-header">📰 محلل الأخبار الذكي</h1>', unsafe_allow_html=True)
st.markdown("---")

# الشريط الجانبي
st.sidebar.title("🎛️ لوحة التحكم")
st.sidebar.markdown("---")

# اختيار نوع التحليل
analysis_type = st.sidebar.selectbox(
    "نوع التحليل",
    ["تصنيف النصوص", "تلخيص النصوص", "استخراج الكيانات", "تحليل شامل"]
)

# خيارات إضافية
st.sidebar.markdown("### ⚙️ الإعدادات")
show_confidence = st.sidebar.checkbox("عرض مستوى الثقة", True)
show_stats = st.sidebar.checkbox("عرض الإحصائيات", True)
show_visualizations = st.sidebar.checkbox("عرض المخططات", True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 حول التطبيق")
st.sidebar.info(
    """
    **المميزات:**
    - تصنيف الأخبار إل