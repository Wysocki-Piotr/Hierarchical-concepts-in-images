"""
Utility functions for model training and evaluation.
"""

import contextlib
import joblib
from tqdm import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to show progress bar with joblib.parallel_backend.
    
    Usage:
        with tqdm_joblib(tqdm(desc='Processing', total=n_jobs)) as progress_bar:
            results = joblib.Parallel(n_jobs=-1)(...)
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    
    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
