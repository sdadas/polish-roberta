from typing import List, Dict, Set
import io

class TableColumn(object):
    ALIGN_RIGHT = '<'
    ALIGN_LEFT = ">"
    ALIGN_CENTER = "^"

    def __init__(self, align: str=ALIGN_LEFT, width: int=None):
        self.align: str = align
        self.width: int = width
        self.name: str = ""
        self.values: Set[str] = set()

    def value(self, value: str):
        self.values.add(value)

    def maxlen(self) -> int:
        return max([len(val) for val in self.values] + [len(self.name)])

    def cell_format(self, align: str=None) -> str:
        aligned: str = align if align is not None else self.align
        width: int = self.width if self.width is not None else self.maxlen()
        return f"{{:{aligned}{width}}}"

    def multiple_values(self) -> bool:
        return len(self.values) > 1


class TablePrinter(object):

    def __init__(self, paddding: int=2):
        self.padding: int = paddding
        self.columns: Dict[int, TableColumn] = {}

    def column(self, idx: int, align: str=TableColumn.ALIGN_LEFT, width: int=None):
        self.columns[idx] = TableColumn(align, width)

    def __get_column(self, idx: int):
        res = self.columns.get(idx, TableColumn())
        self.columns[idx] = res
        return res

    def __prepare_columns(self, table: List[List[str]], includes_header: bool):
        for row_idx, row in enumerate(table):
            if includes_header and row_idx == 0:
                for col_idx, col in enumerate(row):
                    column: TableColumn = self.__get_column(col_idx)
                    column.name = col
                continue
            for col_idx, col in enumerate(row):
                column: TableColumn = self.__get_column(col_idx)
                column.value(col)

    def __print_table(self, table: List[List[str]], includes_header: bool, ignore_single_values: bool):
        pad = " " * self.padding
        res: List[str] = []
        for row_idx, row in enumerate(table):
            row_string = io.StringIO()
            row_string.write(pad)
            for col_idx, col in enumerate(row):
                column: TableColumn = self.__get_column(col_idx)
                if ignore_single_values and not column.multiple_values(): continue
                align = TableColumn.ALIGN_CENTER if row_idx == 0 and includes_header else None
                row_string.write(column.cell_format(align=align).format(col))
                row_string.write(pad)
            res.append(row_string.getvalue())
            row_string.close()
        if includes_header and len(res) > 0:
            separator = "=" * len(res[0])
            res.insert(0, separator)
            res.insert(2, separator)
            res.append(separator)
        for row in res: print(row)

    def print(self, table: List[List[str]], includes_header: bool=True, ignore_single_values: bool=False):
        self.__prepare_columns(table, includes_header)
        self.__print_table(table, includes_header, ignore_single_values)
