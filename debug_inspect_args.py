
import inspect
try:
    from inspect_evals.bigcodebench import bigcodebench
    print(f"Signature of bigcodebench: {inspect.signature(bigcodebench)}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
