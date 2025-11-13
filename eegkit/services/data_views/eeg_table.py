from __future__ import annotations

import pandas as pd

from ...models import BaseTaskDTO, TableInfoDTO
from . import register_data


@register_data("EEG Table", TableInfoDTO)
def show_table(self, task_dto: BaseTaskDTO, table_info: TableInfoDTO):
    task_model = self.get_task(task_dto)
    df_map = {
        'events': task_model.get_event(),
        'channels': task_model.channels,
        'electrodes': task_model.electrodes
    }
    return df_map.get(table_info.table_type, pd.DataFrame()).head(table_info.rows)
