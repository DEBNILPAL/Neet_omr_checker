"""
NEET OMR Checker - Main Streamlit Application
Checks NEET OMR sheets and calculates scores as per official NEET rules.
"""

import streamlit as st
import numpy as np
import json
import os
import shutil
from PIL import Image
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from omr_processor import process_omr_image, visualize_detection, preprocess_image
from neet_scorer import (
    calculate_neet_score, score_column, answer_index_to_letter, letter_to_index
)
from ai_analyzer import (
    analyze_with_gemini, analyze_with_openrouter, get_available_openrouter_models
)

UPLOAD_DIR = "uploads"
BLANK_OMR_DIR = os.path.join(UPLOAD_DIR, "blank")
ANSWER_KEY_DIR = os.path.join(UPLOAD_DIR, "answer_key")
STUDENT_DIR = os.path.join(UPLOAD_DIR, "student")
ANSWER_KEY_JSON = os.path.join(ANSWER_KEY_DIR, "answer_key.json")
BLANK_OMR_FILE = os.path.join(BLANK_OMR_DIR, "blank_omr.jpg")

for d in [BLANK_OMR_DIR, ANSWER_KEY_DIR, STUDENT_DIR]:
    os.makedirs(d, exist_ok=True)


def load_answer_key():
    """Load saved answer key from file."""
    if os.path.exists(ANSWER_KEY_JSON):
        with open(ANSWER_KEY_JSON, 'r') as f:
            return json.load(f)
    return None


def save_answer_key(answers):
    """Save answer key to file."""
    with open(ANSWER_KEY_JSON, 'w') as f:
        json.dump(answers, f)


def save_blank_omr(file):
    """Save blank OMR file."""
    with open(BLANK_OMR_FILE, 'wb') as f:
        f.write(file.getvalue())


def get_blank_omr():
    """Load blank OMR file if exists."""
    if os.path.exists(BLANK_OMR_FILE):
        return BLANK_OMR_FILE
    if os.path.exists("sample_omr.jpeg"):
        return "sample_omr.jpeg"
    return None


def render_answer_table(answers_dict, title="Answers"):
    """Render answer table for 4 columns."""
    col_data = {}
    for col_num in range(1, 5):
        col_key = f"col_{col_num}"
        answers = answers_dict.get(col_key, [-1] * 50)
        col_data[f"Col {col_num} (Q{(col_num-1)*50+1}-{col_num*50})"] = [
            answer_index_to_letter(a) for a in answers
        ]
    
    df = pd.DataFrame(col_data, index=range(1, 51))
    df.index.name = "Q#"
    st.dataframe(df, use_container_width=True, height=400)


def plot_score_breakdown(report):
    """Create score breakdown chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    col_marks = [col['total_marks'] for col in report['columns']]
    col_labels = ['Physics\n(Col 1)', 'Chemistry\n(Col 2)', 'Botany\n(Col 3)', 'Zoology\n(Col 4)']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    
    axes[0].bar(col_labels, col_marks, color=colors, edgecolor='white', linewidth=1.5)
    axes[0].set_title('Marks by Subject', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Marks')
    axes[0].axhline(y=0, color='black', linewidth=0.8)
    for i, (v, c) in enumerate(zip(col_marks, col_labels)):
        axes[0].text(i, v + (1 if v >= 0 else -3), str(v), ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')
    
    correct = report['total_correct']
    wrong = report['total_wrong']
    unattempted = report['total_unattempted']
    
    labels = [f'Correct\n({correct})', f'Wrong\n({wrong})', f'Unattempted\n({unattempted})']
    sizes = [max(correct, 0.1), max(wrong, 0.1), max(unattempted, 0.1)]
    pie_colors = ['#4CAF50', '#F44336', '#9E9E9E']
    explode = (0.05, 0.05, 0.05)
    
    axes[1].pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%',
                explode=explode, startangle=90, textprops={'fontsize': 11})
    axes[1].set_title('Answer Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_question_heatmap(report):
    """Create question-by-question heatmap."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 12))
    
    subject_names = ['Physics', 'Chemistry', 'Botany', 'Zoology']
    
    for col_idx, (col_result, ax) in enumerate(zip(report['columns'], axes)):
        colors_map = {
            'correct': [0, 0.7, 0],
            'wrong': [0.9, 0, 0],
            'multiple_marked': [1, 0.5, 0],
            'unattempted': [0.85, 0.85, 0.85],
            'not_counted': [0.6, 0.6, 0.9],
        }
        
        data = np.zeros((50, 1, 3))
        
        for q in col_result['questions']:
            q_idx = q['q_num'] - 1 - (col_result['col_num'] - 1) * 0
            status = q['status']
            if status == 'correct':
                color = colors_map['correct']
            elif status in ['wrong', 'multiple_marked']:
                color = colors_map['wrong']
            elif status == 'unattempted':
                color = colors_map['unattempted']
            elif 'not_counted' in status:
                color = colors_map['not_counted']
            else:
                color = [0.85, 0.85, 0.85]
            
            if 0 <= q_idx < 50:
                data[q_idx, 0] = color
        
        ax.imshow(data, aspect='auto')
        ax.set_title(f'{subject_names[col_idx]}\n{col_result["total_marks"]} marks', fontsize=11, fontweight='bold')
        ax.set_yticks(range(0, 50, 5))
        ax.set_yticklabels(range(1, 51, 5), fontsize=8)
        ax.set_xticks([])
        
        ax.axhline(y=34.5, color='white', linewidth=2, linestyle='--')
        ax.text(0.5, 34.5, 'Optional→', color='white', fontsize=7,
                ha='center', va='bottom', transform=ax.get_yaxis_transform())
    
    patches = [
        mpatches.Patch(color='green', label='Correct (+4)'),
        mpatches.Patch(color='red', label='Wrong (-5)'),
        mpatches.Patch(color='#D3D3D3', label='Unattempted'),
        mpatches.Patch(color='#9999EE', label='Not Counted'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=10)
    
    plt.suptitle('Question-by-Question Analysis', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def teacher_panel():
    """Teacher panel for uploading blank OMR and answer key."""
    st.header("Teacher Panel")
    st.markdown("Upload the blank OMR sheet for students to download, and set the answer key.")
    
    with st.expander("Step 1: Upload Blank OMR Sheet", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            blank_file = st.file_uploader(
                "Upload Blank OMR Sheet (for students to download)",
                type=["jpg", "jpeg", "png", "pdf"],
                key="blank_omr_upload"
            )
            
            if blank_file:
                save_blank_omr(blank_file)
                st.success("Blank OMR sheet saved successfully!")
        
        with col2:
            blank_path = get_blank_omr()
            if blank_path:
                st.image(blank_path, caption="Current Blank OMR", use_container_width=True)
                st.download_button(
                    label="Download Blank OMR",
                    data=open(blank_path, 'rb').read(),
                    file_name="blank_omr.jpg",
                    mime="image/jpeg"
                )
    
    st.divider()
    
    with st.expander("Step 2: Set Answer Key", expanded=True):
        st.markdown("""
        **Option A:** Upload filled answer key OMR sheet (AI/CV will read it automatically)
        **Option B:** Enter answers manually
        """)
        
        tab_upload, tab_manual = st.tabs(["Upload Answer Key OMR", "Enter Manually"])
        
        with tab_upload:
            answer_key_file = st.file_uploader(
                "Upload Correctly Answered OMR (Answer Key)",
                type=["jpg", "jpeg", "png"],
                key="answer_key_upload"
            )
            
            if answer_key_file:
                st.image(answer_key_file, caption="Answer Key OMR", use_container_width=True)
                answer_key_file.seek(0)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    use_ai = st.checkbox("Use AI for better accuracy", value=False)
                
                if use_ai:
                    ai_provider = st.selectbox("AI Provider", ["Gemini", "OpenRouter"])
                    api_key = st.text_input(f"{ai_provider} API Key", type="password")
                    
                    if ai_provider == "OpenRouter":
                        model_choice = st.selectbox("Model", get_available_openrouter_models())
                
                if st.button("Process Answer Key OMR", type="primary"):
                    with st.spinner("Processing answer key..."):
                        answer_key_file.seek(0)
                        
                        if use_ai and api_key:
                            answer_key_file.seek(0)
                            pil_img = Image.open(answer_key_file)
                            
                            if ai_provider == "Gemini":
                                answers = analyze_with_gemini(pil_img, api_key)
                            else:
                                answers = analyze_with_openrouter(pil_img, api_key, model_choice)
                            
                            if 'error' in answers:
                                st.error(f"AI Error: {answers['error']}")
                                st.info("Falling back to traditional detection...")
                                answer_key_file.seek(0)
                                answers = process_omr_image(answer_key_file)
                        else:
                            answer_key_file.seek(0)
                            answers = process_omr_image(answer_key_file)
                        
                        save_answer_key(answers)
                        st.session_state['answer_key'] = answers
                        st.success("Answer key processed and saved!")
                        
                        st.markdown("**Detected Answer Key:**")
                        render_answer_table(answers, "Answer Key")
                        
                        answer_key_file.seek(0)
                        img = Image.open(answer_key_file)
                        vis = visualize_detection(img, answers)
                        st.image(vis, caption="Detected Bubbles", use_container_width=True)
        
        with tab_manual:
            st.markdown("Enter answers for each column (A/B/C/D for each question):")
            
            existing_key = load_answer_key()
            
            manual_answers = {}
            
            col1, col2, col3, col4 = st.columns(4)
            cols = [col1, col2, col3, col4]
            subjects = ['Physics', 'Chemistry', 'Botany', 'Zoology']
            
            for col_idx, (col_widget, subject) in enumerate(zip(cols, subjects)):
                with col_widget:
                    st.markdown(f"**{subject}**")
                    col_key = f"col_{col_idx + 1}"
                    col_answers = []
                    
                    for q in range(50):
                        existing = '-'
                        if existing_key and col_key in existing_key:
                            existing = answer_index_to_letter(existing_key[col_key][q]) if q < len(existing_key[col_key]) else '-'
                        
                        default_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, '-': 4}.get(existing, 4)
                        
                        ans = st.selectbox(
                            f"Q{(col_idx * 50) + q + 1}",
                            ['A', 'B', 'C', 'D', '-'],
                            index=default_idx,
                            key=f"manual_ans_{col_idx}_{q}",
                            label_visibility="collapsed"
                        )
                        col_answers.append(letter_to_index(ans) if ans != '-' else -1)
                    
                    manual_answers[col_key] = col_answers
            
            if st.button("Save Manual Answer Key", type="primary"):
                save_answer_key(manual_answers)
                st.session_state['answer_key'] = manual_answers
                st.success("Answer key saved successfully!")
    
    existing_key = load_answer_key()
    if existing_key:
        st.divider()
        st.success("Answer key is set and ready!")
        if st.button("View Current Answer Key"):
            render_answer_table(existing_key, "Current Answer Key")


def student_panel():
    """Student panel for downloading blank OMR and submitting filled OMR."""
    st.header("Student Panel")
    
    answer_key = load_answer_key()
    if not answer_key:
        st.warning("No answer key has been set yet. Please ask your teacher to set it up first.")
        return
    
    with st.expander("Step 1: Download Blank OMR Sheet", expanded=True):
        blank_path = get_blank_omr()
        if blank_path:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(blank_path, caption="Blank OMR Sheet", use_container_width=True)
            with col2:
                st.markdown("""
                ### Instructions:
                1. Download the blank OMR sheet below
                2. Print it on A4 paper
                3. Fill in your Roll Number and other details
                4. Answer the questions by completely darkening the bubbles
                5. **4 columns, 50 questions each** (Physics, Chemistry, Botany, Zoology)
                6. First 35 questions per column are **mandatory**
                7. Questions 36-50 per column: attempt **only 10** (first 10 consecutive)
                8. Upload your completed OMR below
                """)
                
                with open(blank_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Blank OMR Sheet",
                        data=f.read(),
                        file_name="neet_blank_omr.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
        else:
            st.error("No blank OMR sheet available. Please contact your teacher.")
    
    st.divider()
    
    with st.expander("Step 2: Upload Your Completed OMR", expanded=True):
        student_name = st.text_input("Your Name (optional)")
        roll_number = st.text_input("Roll Number (optional)")
        
        student_file = st.file_uploader(
            "Upload Your Completed OMR Sheet",
            type=["jpg", "jpeg", "png"],
            key="student_omr_upload"
        )
        
        if student_file:
            st.image(student_file, caption="Your OMR Sheet", use_container_width=True)
            student_file.seek(0)
            
            st.markdown("### AI Enhancement (Optional)")
            st.markdown("Use AI for more accurate bubble detection:")
            
            use_ai_col1, use_ai_col2 = st.columns(2)
            with use_ai_col1:
                use_ai = st.checkbox("Enable AI Analysis", value=False)
            
            ai_answers = None
            if use_ai:
                ai_provider = st.selectbox("Select AI Provider", ["Gemini", "OpenRouter"], key="student_ai_provider")
                api_key = st.text_input(f"{ai_provider} API Key", type="password", key="student_api_key")
                
                if ai_provider == "OpenRouter":
                    model_choice = st.selectbox("Model", get_available_openrouter_models(), key="student_model")
                else:
                    model_choice = None
                
                if api_key:
                    st.info("AI analysis will run when you click 'Check My OMR'")
            
            if st.button("Check My OMR", type="primary", use_container_width=True):
                with st.spinner("Analyzing your OMR sheet..."):
                    student_file.seek(0)
                    pil_img = Image.open(student_file)
                    
                    cv_answers = process_omr_image(pil_img)
                    
                    final_answers = cv_answers
                    detection_method = "Traditional CV"
                    
                    if use_ai and api_key:
                        with st.spinner("Running AI analysis for better accuracy..."):
                            if ai_provider == "Gemini":
                                ai_answers = analyze_with_gemini(pil_img, api_key)
                            else:
                                ai_answers = analyze_with_openrouter(pil_img, api_key, model_choice)
                            
                            if 'error' not in ai_answers:
                                final_answers = ai_answers
                                detection_method = f"AI ({ai_provider})"
                            else:
                                st.warning(f"AI failed: {ai_answers['error']}. Using traditional detection.")
                    
                    report = calculate_neet_score(final_answers, answer_key)
                    
                    st.session_state['last_report'] = report
                    st.session_state['last_detection'] = final_answers
                    st.session_state['detection_method'] = detection_method
                
                show_results(report, final_answers, detection_method, student_name, roll_number, pil_img)


def show_results(report, detected_answers, method, student_name="", roll_number="", student_image=None):
    """Display comprehensive results."""
    st.divider()
    st.markdown("## Results")
    
    if student_name or roll_number:
        info_parts = []
        if student_name:
            info_parts.append(f"**Student:** {student_name}")
        if roll_number:
            info_parts.append(f"**Roll No:** {roll_number}")
        st.markdown(" | ".join(info_parts))
    
    st.markdown(f"*Detection Method: {method}*")
    
    total_marks = report['total_marks']
    
    if total_marks >= 600:
        grade_color = "#4CAF50"
    elif total_marks >= 450:
        grade_color = "#2196F3"
    elif total_marks >= 300:
        grade_color = "#FF9800"
    else:
        grade_color = "#F44336"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Score", f"{total_marks}/720")
    
    with col2:
        st.metric("Percentage", f"{report['percentage']}%")
    
    with col3:
        st.metric("Correct Answers", report['total_correct'])
    
    with col4:
        st.metric("Wrong Answers", report['total_wrong'])
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0;">
        <h1 style="color: white; font-size: 48px; margin: 0;">{total_marks}</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 18px; margin: 5px 0;">out of 720</p>
        <p style="color: rgba(255,255,255,0.8); font-size: 16px;">{report['grade']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Subject-wise Breakdown")
    subjects = ['Physics', 'Chemistry', 'Botany', 'Zoology']
    
    for col_idx, (col_result, subject) in enumerate(zip(report['columns'], subjects)):
        exp_label = f"**{subject}** (Col {col_result['col_num']}) — {col_result['total_marks']} marks"
        
        with st.expander(exp_label):
            sub_col1, sub_col2, sub_col3 = st.columns(3)
            
            with sub_col1:
                st.metric("Mandatory Section", 
                         f"{col_result['mandatory_correct']} correct",
                         f"{col_result['mandatory_wrong']} wrong")
            
            with sub_col2:
                st.metric("Mandatory Unattempted", col_result['mandatory_unattempted'])
            
            with sub_col3:
                st.metric("Optional Counted", 
                         f"{col_result['optional_counted']}/10",
                         f"{col_result['optional_correct']} correct")
            
            q_data = []
            for q in col_result['questions']:
                q_num_display = q['q_num']
                status_emoji = {
                    'correct': '✅',
                    'wrong': '❌',
                    'unattempted': '⬜',
                    'multiple_marked': '⚠️',
                    'not_counted': '🔵',
                    'multiple_marked_not_counted': '🔵',
                }.get(q['status'], '❓')
                
                section = 'Mandatory' if q['section'] == 'mandatory' else 'Optional'
                counted = '✓' if q.get('counted', True) else '✗ (not counted)'
                
                q_data.append({
                    'Q#': q_num_display,
                    'Section': section,
                    'Your Answer': answer_index_to_letter(q['student_answer']),
                    'Correct': answer_index_to_letter(q['correct_answer']),
                    'Status': f"{status_emoji} {q['status'].replace('_', ' ').title()}",
                    'Marks': q['marks'],
                    'Counted': counted
                })
            
            df = pd.DataFrame(q_data)
            st.dataframe(df, use_container_width=True, height=300)
    
    st.subheader("Visual Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Score Chart", "Question Heatmap", "Detected Answers"])
    
    with tab1:
        fig = plot_score_breakdown(report)
        st.pyplot(fig)
        plt.close()
    
    with tab2:
        fig2 = plot_question_heatmap(report)
        st.pyplot(fig2)
        plt.close()
    
    with tab3:
        if student_image:
            vis_img = visualize_detection(student_image, detected_answers)
            st.image(vis_img, caption="Detected Bubbles (Green=detected)", use_container_width=True)
        st.markdown("**Detected Answers:**")
        render_answer_table(detected_answers, "Detected Answers")
    
    report_text = generate_report_text(report, subjects, student_name, roll_number)
    st.download_button(
        "📄 Download Detailed Report",
        data=report_text,
        file_name=f"neet_result_{roll_number or 'student'}.txt",
        mime="text/plain"
    )


def generate_report_text(report, subjects, student_name, roll_number):
    """Generate a text report for download."""
    lines = [
        "=" * 60,
        "NEET OMR CHECKER - RESULT REPORT",
        "=" * 60,
        f"Student: {student_name or 'N/A'}",
        f"Roll Number: {roll_number or 'N/A'}",
        "",
        f"TOTAL SCORE: {report['total_marks']} / 720",
        f"Percentage: {report['percentage']}%",
        f"Grade: {report['grade']}",
        f"Total Correct: {report['total_correct']}",
        f"Total Wrong: {report['total_wrong']}",
        f"Total Unattempted: {report['total_unattempted']}",
        "",
        "-" * 60,
        "SUBJECT-WISE BREAKDOWN",
        "-" * 60,
    ]
    
    for col_result, subject in zip(report['columns'], subjects):
        lines.extend([
            f"\n{subject} (Column {col_result['col_num']}): {col_result['total_marks']} marks",
            f"  Mandatory Section (Q1-35):",
            f"    Correct: {col_result['mandatory_correct']}",
            f"    Wrong: {col_result['mandatory_wrong']}",
            f"    Unattempted: {col_result['mandatory_unattempted']}",
            f"  Optional Section (Q36-50, only 10 counted):",
            f"    Correct: {col_result['optional_correct']}",
            f"    Wrong: {col_result['optional_wrong']}",
            f"    Counted: {col_result['optional_counted']}",
        ])
    
    lines.extend([
        "",
        "-" * 60,
        "QUESTION-BY-QUESTION DETAIL",
        "-" * 60,
    ])
    
    for col_result, subject in zip(report['columns'], subjects):
        lines.append(f"\n{subject}:")
        lines.append(f"{'Q#':<5} {'Section':<12} {'Your Ans':<10} {'Correct':<10} {'Status':<20} {'Marks'}")
        
        for q in col_result['questions']:
            lines.append(
                f"{q['q_num']:<5} {q['section']:<12} "
                f"{answer_index_to_letter(q['student_answer']):<10} "
                f"{answer_index_to_letter(q['correct_answer']):<10} "
                f"{q['status']:<20} {q['marks']}"
            )
    
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def about_panel():
    """About/Help panel."""
    st.header("About & Help")
    
    st.markdown("""
    ## NEET OMR Checker
    
    This application helps evaluate NEET (National Eligibility cum Entrance Test) OMR sheets automatically.
    
    ### Scoring Rules (as per NEET)
    
    | Section | Questions | Correct | Wrong | Unattempted |
    |---------|-----------|---------|-------|-------------|
    | Mandatory | Q1-35 (per column) | +4 marks | -5 marks | 0 marks |
    | Optional | Q36-50 (per column) | +4 marks | -5 marks | 0 marks |
    
    **Optional Section Rules:**
    - Only 10 questions should be attempted
    - If more than 10 are attempted, only the **first 10 consecutive** answers are counted
    - The remaining answers are ignored
    
    ### Column Structure
    | Column | Subject | Questions |
    |--------|---------|-----------|
    | Column 1 | Physics | Q1-Q50 |
    | Column 2 | Chemistry | Q51-Q100 |
    | Column 3 | Botany | Q101-Q150 |
    | Column 4 | Zoology | Q151-Q200 |
    
    **Maximum Score: 720 marks** (180 questions × 4 marks)
    
    ### Detection Methods
    
    **Traditional CV (Computer Vision):**
    - Uses OpenCV for bubble detection
    - Works offline, no API key needed
    - Best for clearly filled, high-quality scans
    
    **AI-Enhanced Detection:**
    - Uses Gemini or OpenRouter vision models
    - Better accuracy for partially filled or poor-quality images
    - Requires an API key
    
    ### Getting API Keys
    
    **Gemini API:**
    1. Visit [Google AI Studio](https://aistudio.google.com/)
    2. Create a free API key
    
    **OpenRouter API:**
    1. Visit [OpenRouter.ai](https://openrouter.ai/)
    2. Sign up and get API key
    3. Multiple AI models available (GPT-4o, Claude, Gemini, etc.)
    
    ### Tips for Best Results
    - Use good lighting when scanning/photographing the OMR
    - Ensure the entire OMR sheet is visible in the image
    - Fill bubbles completely and darkly
    - Avoid stray marks outside bubbles
    - Use A4 size paper for printing
    """)


def main():
    st.set_page_config(
        page_title="NEET OMR Checker",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("NEET OMR Checker")
    st.markdown("*Automated OMR checking with AI-enhanced accuracy for NEET exam sheets*")
    
    with st.sidebar:
        st.image("sample_omr.jpeg" if os.path.exists("sample_omr.jpeg") else [], 
                 use_container_width=True)
        st.markdown("---")
        st.markdown("### Navigation")
        
        page = st.radio(
            "Select Panel",
            ["Teacher Panel", "Student Panel", "About & Help"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        answer_key = load_answer_key()
        blank_exists = get_blank_omr() is not None
        
        st.markdown("### System Status")
        st.markdown(f"{'✅' if blank_exists else '❌'} Blank OMR: {'Ready' if blank_exists else 'Not set'}")
        st.markdown(f"{'✅' if answer_key else '❌'} Answer Key: {'Set' if answer_key else 'Not set'}")
        
        st.markdown("---")
        st.markdown("### NEET Scoring Rules")
        st.markdown("""
        - **Correct:** +4 marks
        - **Wrong:** -5 marks  
        - **Unattempted:** 0 marks
        - **Q1-35:** Mandatory
        - **Q36-50:** Optional (10 max)
        - **Max Score:** 720
        """)
    
    if page == "Teacher Panel":
        teacher_panel()
    elif page == "Student Panel":
        student_panel()
    else:
        about_panel()


if __name__ == "__main__":
    main()
