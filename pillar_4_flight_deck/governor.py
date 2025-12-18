"""
Agent Execution Governor - Production reliability patterns
From: The Architect's Playbook, Pillar IV
"""

import time
import random
import logging
import threading
import concurrent.futures
from datetime import datetime
from typing import Callable, Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when execution exceeds budget"""
    pass


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


@dataclass
class ExecutionResult:
    success: bool
    result: Any
    duration: float
    cost: float
    retries: int
    error: Optional[str] = None


class CostTracker:
    """Track token usage and costs across agent execution"""
    
    PRICING = {
        'gpt-4o': {'input': 2.50, 'output': 10.00},
        'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
        'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
    }
    
    def __init__(self, max_cost: float):
        self.max_cost = max_cost
        self.current_cost = 0.0
        self.call_log: List[Dict] = []
    
    def record_call(self, model: str, input_tokens: int, output_tokens: int):
        pricing = self.PRICING.get(model, {'input': 0, 'output': 0})
        cost = (input_tokens / 1_000_000) * pricing['input'] + \
               (output_tokens / 1_000_000) * pricing['output']
        
        self.current_cost += cost
        self.call_log.append({'model': model, 'cost': cost, 'timestamp': datetime.now()})
        
        if self.current_cost > self.max_cost:
            raise BudgetExceededError(f"Cost ${self.current_cost:.4f} exceeds ${self.max_cost}")
    
    def get_summary(self) -> Dict:
        return {
            'total_cost': self.current_cost,
            'budget': self.max_cost,
            'remaining': self.max_cost - self.current_cost
        }


class RetryPolicy:
    """Exponential backoff with jitter"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == self.max_retries:
                    break
                
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                delay *= 0.5 + random.random()  # Jitter
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                time.sleep(delay)
        
        raise last_exception


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Prevent cascading failures"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, success_threshold: int = 2):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
    
    def allow_request(self) -> bool:
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if self._timeout_elapsed():
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False
            return True
    
    def record_success(self):
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self._reset()
            else:
                self.failure_count = 0
    
    def record_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPEN")
    
    def _timeout_elapsed(self) -> bool:
        if not self.last_failure_time:
            return True
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.timeout
    
    def _reset(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info("Circuit breaker CLOSED")


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, tokens: int = 1, timeout: float = None) -> bool:
        start = time.time()
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
            if timeout and (time.time() - start) >= timeout:
                return False
            time.sleep(0.1)
    
    def _refill(self):
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.last_update) * self.rate)
        self.last_update = now


class AgentExecutionGovernor:
    """Complete execution governance for production agents"""
    
    def __init__(
        self,
        max_cost_per_execution: float = 1.0,
        max_execution_time: int = 300,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
        rate_limit: float = 10.0
    ):
        self.cost_tracker = CostTracker(max_cost_per_execution)
        self.timeout_seconds = max_execution_time
        self.retry_policy = RetryPolicy(max_retries=max_retries)
        self.circuit_breaker = CircuitBreaker(failure_threshold=circuit_breaker_threshold)
        self.rate_limiter = RateLimiter(rate=rate_limit, capacity=100)
    
    def execute(self, agent_func: Callable, *args, **kwargs) -> ExecutionResult:
        start_time = time.time()
        
        if not self.circuit_breaker.allow_request():
            return ExecutionResult(False, None, 0, 0, 0, "Circuit breaker OPEN")
        
        if not self.rate_limiter.acquire(timeout=5.0):
            return ExecutionResult(False, None, 0, 0, 0, "Rate limit exceeded")
        
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: self.retry_policy.execute(agent_func, *args, **kwargs)
                )
                result = future.result(timeout=self.timeout_seconds)
            
            self.circuit_breaker.record_success()
            return ExecutionResult(
                success=True,
                result=result,
                duration=time.time() - start_time,
                cost=self.cost_tracker.current_cost,
                retries=0
            )
        except concurrent.futures.TimeoutError:
            self.circuit_breaker.record_failure()
            return ExecutionResult(False, None, self.timeout_seconds, 0, 0, "Timeout")
        except BudgetExceededError as e:
            return ExecutionResult(False, None, time.time() - start_time, 
                                   self.cost_tracker.current_cost, 0, str(e))
        except Exception as e:
            self.circuit_breaker.record_failure()
            return ExecutionResult(False, None, time.time() - start_time, 0, 0, str(e))
