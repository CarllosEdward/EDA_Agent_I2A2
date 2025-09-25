from .data_loading_task import create_data_loading_task
from .analysis_task import create_analysis_task
from .visualization_task import create_visualization_task
from .conclusion_task import create_conclusion_task

__all__ = [
    'create_data_loading_task',
    'create_analysis_task',
    'create_visualization_task', 
    'create_conclusion_task'
]
