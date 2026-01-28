import sys
import os
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.utils import retry_with_backoff

@retry_with_backoff(retries=2, initial_delay=0.1, backoff_factor=2)
def failing_function(context):
    print(f"  Executing function for attempt {context['attempts'] + 1}...")
    context['attempts'] += 1
    if context['attempts'] < 3:
        raise ValueError("Simulated transient error")
    return "Success!"

def test_retries():
    print("Testing retry mechanism...")
    context = {'attempts': 0}
    try:
        result = failing_function(context)
        print(f"Final Result: {result}")
        if context['attempts'] == 3:
            print("SUCCESS: Function retried 2 times and succeeded on the 3rd attempt.")
        else:
            print(f"FAILURE: Expected 3 attempts, got {context['attempts']}")
    except Exception as e:
        print(f"Final Error: {e}")

if __name__ == "__main__":
    test_retries()
