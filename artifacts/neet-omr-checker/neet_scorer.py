"""
NEET Scoring Module
Implements official NEET exam scoring rules.

Rules:
- 4 columns, 50 questions each = 200 total questions
- Max marks = 720 (180 questions × 4)

Per column scoring:
- First 35 questions: MANDATORY
    - Attempted correct: +4
    - Attempted wrong: -5 (negative marking for NEET is actually -1, but per user spec: -5)
    - Not attempted: 0 marks (mandatory means must attempt, but if not attempted -> 0)
  Wait - user said "if any one not attempts then no marks given to student"
  and "if any question wrong then -5 will be count"
  
  Reread: "first 35 surely attempts if any one not attempts then no marks given to student"
  -> if student doesn't attempt a mandatory question, that question gives 0 marks
  -> if wrong: -5

- Last 15 questions (Q36-Q50): Optional pool
    - Only 10 questions should be attempted
    - If more than 10 attempted, consider first consecutive 10
    - Wrong answer: -5
    - Correct: +4
    - More than 10: only count first 10 consecutive answers
"""

OPTIONS = ['A', 'B', 'C', 'D']


def score_column(student_answers, correct_answers, col_num):
    """
    Score a single column based on NEET rules.
    
    Args:
        student_answers: list of 50 answers (-1=unattempted, -2=multiple, 0-3=A-D)
        correct_answers: list of 50 answers (0-3 for A-D)
        col_num: column number (1-4) for display
    
    Returns:
        dict with detailed scoring info
    """
    results = {
        'col_num': col_num,
        'questions': [],
        'mandatory_correct': 0,
        'mandatory_wrong': 0,
        'mandatory_unattempted': 0,
        'optional_correct': 0,
        'optional_wrong': 0,
        'optional_counted': 0,
        'optional_skipped': 0,
        'total_marks': 0,
        'max_possible': 0,
    }
    
    mandatory_marks = 0
    optional_marks = 0
    
    mandatory_section = list(zip(range(35), student_answers[:35], correct_answers[:35]))
    optional_section = list(zip(range(35, 50), student_answers[35:50], correct_answers[35:50]))
    
    for q_idx, s_ans, c_ans in mandatory_section:
        q_result = {
            'q_num': q_idx + 1,
            'section': 'mandatory',
            'student_answer': s_ans,
            'correct_answer': c_ans,
            'marks': 0,
            'status': '',
            'counted': True
        }
        
        if s_ans == -1:
            q_result['marks'] = 0
            q_result['status'] = 'unattempted'
            results['mandatory_unattempted'] += 1
        elif s_ans == -2:
            q_result['marks'] = -5
            q_result['status'] = 'multiple_marked'
            results['mandatory_wrong'] += 1
        elif s_ans == c_ans:
            q_result['marks'] = 4
            q_result['status'] = 'correct'
            results['mandatory_correct'] += 1
        else:
            q_result['marks'] = -5
            q_result['status'] = 'wrong'
            results['mandatory_wrong'] += 1
        
        mandatory_marks += q_result['marks']
        results['questions'].append(q_result)
    
    results['max_possible'] = 35 * 4
    
    attempted_optional = []
    for q_idx, s_ans, c_ans in optional_section:
        if s_ans != -1:
            attempted_optional.append((q_idx, s_ans, c_ans))
    
    if len(attempted_optional) > 10:
        counted_optional = []
        all_15 = optional_section
        consecutive_count = 0
        consecutive_attempts = []
        
        for item in all_15:
            q_idx, s_ans, c_ans = item
            if s_ans != -1:
                consecutive_count += 1
                consecutive_attempts.append(item)
                if consecutive_count == 10:
                    break
        
        counted_optional = consecutive_attempts[:10]
        counted_set = set(q_idx for q_idx, _, _ in counted_optional)
    else:
        counted_set = set(q_idx for q_idx, _, _ in attempted_optional)
    
    for q_idx, s_ans, c_ans in optional_section:
        q_result = {
            'q_num': q_idx + 1,
            'section': 'optional',
            'student_answer': s_ans,
            'correct_answer': c_ans,
            'marks': 0,
            'status': '',
            'counted': q_idx in counted_set
        }
        
        if s_ans == -1:
            q_result['status'] = 'unattempted'
            results['optional_skipped'] += 1
        elif q_idx not in counted_set:
            q_result['status'] = 'not_counted'
            if s_ans == -2:
                q_result['status'] = 'multiple_marked_not_counted'
        elif s_ans == -2:
            q_result['marks'] = -5
            q_result['status'] = 'multiple_marked'
            results['optional_wrong'] += 1
            results['optional_counted'] += 1
            optional_marks += q_result['marks']
        elif s_ans == c_ans:
            q_result['marks'] = 4
            q_result['status'] = 'correct'
            results['optional_correct'] += 1
            results['optional_counted'] += 1
            optional_marks += q_result['marks']
        else:
            q_result['marks'] = -5
            q_result['status'] = 'wrong'
            results['optional_wrong'] += 1
            results['optional_counted'] += 1
            optional_marks += q_result['marks']
        
        results['questions'].append(q_result)
    
    results['max_possible'] += 10 * 4
    results['total_marks'] = mandatory_marks + optional_marks
    
    return results


def calculate_neet_score(student_all_answers, correct_all_answers):
    """
    Calculate total NEET score across all 4 columns.
    
    Args:
        student_all_answers: dict with col_1..col_4 each having 50 answers
        correct_all_answers: dict with col_1..col_4 each having 50 answers
    
    Returns:
        Complete scoring report
    """
    report = {
        'columns': [],
        'total_marks': 0,
        'max_marks': 720,
        'total_correct': 0,
        'total_wrong': 0,
        'total_unattempted': 0,
        'percentage': 0.0,
        'grade': '',
    }
    
    total = 0
    for col_num in range(1, 5):
        col_key = f"col_{col_num}"
        s_ans = student_all_answers.get(col_key, [-1] * 50)
        c_ans = correct_all_answers.get(col_key, [0] * 50)
        
        col_result = score_column(s_ans, c_ans, col_num)
        report['columns'].append(col_result)
        total += col_result['total_marks']
        
        report['total_correct'] += col_result['mandatory_correct'] + col_result['optional_correct']
        report['total_wrong'] += col_result['mandatory_wrong'] + col_result['optional_wrong']
        report['total_unattempted'] += col_result['mandatory_unattempted'] + col_result['optional_skipped']
    
    report['total_marks'] = max(0, total)
    report['percentage'] = round((report['total_marks'] / 720) * 100, 2)
    
    pct = report['percentage']
    if pct >= 90:
        report['grade'] = 'Excellent (Top Tier)'
    elif pct >= 80:
        report['grade'] = 'Very Good'
    elif pct >= 70:
        report['grade'] = 'Good'
    elif pct >= 60:
        report['grade'] = 'Average'
    elif pct >= 50:
        report['grade'] = 'Below Average'
    else:
        report['grade'] = 'Needs Improvement'
    
    return report


def answer_index_to_letter(idx):
    """Convert 0-3 index to A-D letter."""
    if idx == -1:
        return '-'
    if idx == -2:
        return 'MULTI'
    letters = ['A', 'B', 'C', 'D']
    if 0 <= idx < 4:
        return letters[idx]
    return '?'


def letter_to_index(letter):
    """Convert A-D letter to 0-3 index."""
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    return mapping.get(str(letter).upper(), -1)
