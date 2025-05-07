"""
Portfolio Allocation System - Usability Testing Guide

This script provides a structured approach to testing the usability of the
portfolio allocation dashboard and models. Use this guide when conducting
usability tests with participants.

Instructions for Test Facilitator:
1. Ensure the Flask app is running at http://localhost:5000
2. Provide this script to participants or use it as a guide while observing them
3. For each task, record:
   - Task completion time
   - Success/failure
   - Any difficulties encountered
   - Participant comments

Metrics to collect:
- Task completion rate
- Time on task
- Error rate
- User satisfaction (1-5 scale)
"""

# Define test tasks
tasks = [
    {
        "task_id": 1,
        "description": "Open the dashboard and identify the current portfolio value and annualized return.",
        "success_criteria": "User correctly identifies the portfolio value and annualized return displayed on the dashboard.",
        "expected_time": "30 seconds"
    },
    {
        "task_id": 2,
        "description": "Run the Mean-Variance Optimization model with a risk aversion parameter of 3.0.",
        "success_criteria": "User selects MVO model, sets risk aversion to 3.0, runs the model, and observes the results.",
        "expected_time": "1 minute"
    },
    {
        "task_id": 3,
        "description": "Compare the performance of different models in terms of Sharpe ratio.",
        "success_criteria": "User navigates to the model comparison section and identifies which model has the highest Sharpe ratio.",
        "expected_time": "45 seconds"
    },
    {
        "task_id": 4,
        "description": "Identify the current allocation between stocks and bonds.",
        "success_criteria": "User correctly identifies the current allocation percentages displayed on the dashboard.",
        "expected_time": "30 seconds"
    },
    {
        "task_id": 5,
        "description": "Run the Deep Reinforcement Learning model and explain what happens during processing.",
        "success_criteria": "User successfully initiates the Deep RL model and can describe the progress indicators.",
        "expected_time": "1 minute"
    }
]

# Post-test questionnaire
questionnaire = [
    {
        "question_id": 1,
        "question": "On a scale of 1-5, how easy was it to navigate the dashboard?",
        "type": "rating"
    },
    {
        "question_id": 2,
        "question": "On a scale of 1-5, how clear was the presentation of the portfolio metrics?",
        "type": "rating"
    },
    {
        "question_id": 3,
        "question": "On a scale of 1-5, how intuitive was the process of running a model?",
        "type": "rating"
    },
    {
        "question_id": 4,
        "question": "What aspects of the interface did you find most helpful?",
        "type": "open"
    },
    {
        "question_id": 5,
        "question": "What aspects of the interface were confusing or difficult to use?",
        "type": "open"
    },
    {
        "question_id": 6,
        "question": "What additional features or information would you like to see in the dashboard?",
        "type": "open"
    }
]

# Template for recording results
result_template = {
    "participant_id": "",
    "date": "",
    "task_results": [],
    "questionnaire_results": []
}

def print_tasks():
    """Print all tasks for the usability test"""
    print("\n=== Usability Test Tasks ===\n")
    for task in tasks:
        print(f"Task {task['task_id']}: {task['description']}")
        print(f"Success criteria: {task['success_criteria']}")
        print(f"Expected time: {task['expected_time']}")
        print()

def print_questionnaire():
    """Print the post-test questionnaire"""
    print("\n=== Post-Test Questionnaire ===\n")
    for q in questionnaire:
        print(f"Q{q['question_id']}: {q['question']}")
        if q['type'] == 'rating':
            print("   (1-5 scale, where 1 = Very difficult/unclear and 5 = Very easy/clear)")
        print()

if __name__ == "__main__":
    print("Portfolio Allocation System - Usability Testing Guide")
    print("=" * 60)
    print("\nThis script provides a structured approach to testing the usability of the")
    print("portfolio allocation dashboard and models.")
    
    print_tasks()
    print_questionnaire()
    
    print("\n=== Instructions for Recording Results ===\n")
    print("For each task, record:")
    print("1. Time taken to complete")
    print("2. Success (Y/N)")
    print("3. Notes on any difficulties observed")
    print("4. Direct quotes from participant")
    print("\nFor the questionnaire, record the numerical rating (1-5) for rating questions")
    print("and full responses for open questions.")