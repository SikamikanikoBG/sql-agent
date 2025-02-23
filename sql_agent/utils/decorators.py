import time
from functools import wraps
from typing import Any, Callable

def prevent_rerun(timeout: int = 60):
    """Decorator to prevent re-running a function within the specified timeout period.
    
    Args:
        timeout (int): Time in seconds to cache the result
    """
    def decorator(func: Callable) -> Callable:
        last_run = {}
        last_result = {}
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_time = time.time()
            # Convert args to a hashable format
            hashable_args = tuple(tuple(arg) if isinstance(arg, list) else arg for arg in args)
            func_key = (func.__name__, hashable_args, frozenset(kwargs.items()))
            
            if func_key in last_run:
                if current_time - last_run[func_key] < timeout:
                    return last_result[func_key]
            
            result = func(*args, **kwargs)
            last_run[func_key] = current_time
            last_result[func_key] = result
            return result
            
        return wrapper
    return decorator
